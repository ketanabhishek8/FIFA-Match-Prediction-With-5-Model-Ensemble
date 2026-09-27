"""Stop whatever the app left running: the FastAPI backend and the Vite frontend.

    python stop.py                  # stop both
    python stop.py --backend-only   # just the API on :8000
    python stop.py --frontend-only  # just the UI on :5173
    python stop.py --dry-run        # list what would be killed, kill nothing

Kills by *port*, not by a remembered PID, so it works when the terminal that
ran start.py is gone - which is the normal case: start.py exits straight away
("Everything was already running - nothing to manage") whenever both ports are
already taken, so after the first launch there's no start.py left to Ctrl+C.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import signal
import subprocess
import sys
import time

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
IS_WINDOWS = sys.platform == "win32"
# How long to wait for a port to actually come free after the kill.
RELEASE_TIMEOUT = 10

# The process holding a port is a *leaf*: `npm run dev` goes npm.cmd -> node
# (npm-cli) -> cmd.exe -> node (vite), and `python -m uvicorn` can re-exec into
# a second python. Killing only the listener leaves those wrappers behind, so
# we walk up while the parent is still one of those wrappers.
#
# Deliberately no `start.py` here. Matching it would walk into the supervisor
# and, worse, into whatever shell launched it (a `bash -c "python start.py"`
# command line matches too) - killing the user's terminal. Stopping at the
# server process is also the *better* outcome: start.py's watch loop notices a
# child died and tears the rest down by itself.
WRAPPER = re.compile(r"uvicorn|api\.main:app|vite|npm", re.IGNORECASE)
# Belt and braces: never walk into a shell/terminal even if it somehow matches.
NEVER_WALK = re.compile(
    r"(?:ba|z|k)?sh(?:\.exe)?|powershell|pwsh|WindowsTerminal|conhost",
    re.IGNORECASE,
)
MAX_WALK = 10  # cycle/PID-reuse guard while walking parents


def run(cmd: list[str]) -> str:
    """Run a helper command, returning stdout ('' if the tool isn't there)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def process_table() -> dict[int, tuple[int, str]]:
    """Map pid -> (parent pid, command line) for every running process."""
    table: dict[int, tuple[int, str]] = {}
    if IS_WINDOWS:
        out = run([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,CommandLine | "
            "ConvertTo-Csv -NoTypeInformation",
        ])
        for row in csv.DictReader(io.StringIO(out)):
            try:
                pid = int(row["ProcessId"])
                ppid = int(row["ParentProcessId"])
            except (TypeError, ValueError, KeyError):
                continue
            table[pid] = (ppid, row.get("CommandLine") or "")
    else:
        for line in run(["ps", "-eo", "pid=,ppid=,args="]).splitlines():
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                table[int(parts[0])] = (int(parts[1]),
                                        parts[2] if len(parts) > 2 else "")
            except ValueError:
                continue
    return table


def listening_pids(port: int) -> set[int]:
    """PIDs listening on localhost:port, across every address family."""
    pids: set[int] = set()
    if IS_WINDOWS:
        # No `-p TCP`: on Windows that filters to IPv4 only, and Vite listens on
        # [::1] while uvicorn listens on 127.0.0.1 - the same address-family trap
        # start.py's port_in_use() documents. Plain `-ano` lists both.
        for line in run(["netstat", "-ano"]).splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0] != "TCP" or "LISTENING" not in fields:
                continue
            local = fields[1]
            if ":" not in local or local.rsplit(":", 1)[1] != str(port):
                continue
            try:
                pids.add(int(fields[-1]))
            except ValueError:
                continue
    else:
        for line in run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"]).split():
            try:
                pids.add(int(line))
            except ValueError:
                continue
    return pids


def kill_root(pid: int, table: dict[int, tuple[int, str]]) -> int:
    """Topmost ancestor of pid that still looks like part of this app."""
    root = pid
    seen = {pid}
    for _ in range(MAX_WALK):
        parent = table.get(root, (0, ""))[0]
        if parent in (0, 1) or parent in seen:
            break
        parent_cmd = table.get(parent, (0, ""))[1]
        if NEVER_WALK.search(parent_cmd) or not WRAPPER.search(parent_cmd):
            break
        seen.add(parent)
        root = parent
    return root


def describe(pid: int, table: dict[int, tuple[int, str]]) -> str:
    cmd = " ".join(table.get(pid, (0, ""))[1].split()) or "<unknown>"
    return f"pid {pid}  {cmd[:110]}"


def terminate(pid: int) -> None:
    if IS_WINDOWS:
        # /T so the whole wrapper tree goes with it; /F because npm's cmd.exe
        # shim ignores a polite close and would sit on "Terminate batch job?".
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def force(pid: int) -> None:
    if IS_WINDOWS:
        return  # taskkill /F was already the hard kill
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def stop_port(port: int, label: str, dry_run: bool) -> bool:
    """Free one port. Returns False only if something is still holding it."""
    pids = listening_pids(port)
    if not pids:
        print(f"[ok] {label}: nothing listening on :{port}")
        return True

    table = process_table()
    roots = {kill_root(pid, table) for pid in pids}
    for root in sorted(roots):
        print(f"[..] {label}: {'would kill' if dry_run else 'killing'} "
              f"{describe(root, table)}")
    if dry_run:
        return True

    for root in roots:
        terminate(root)

    deadline = time.time() + RELEASE_TIMEOUT
    forced = False
    while time.time() < deadline:
        if not listening_pids(port):
            print(f"[PASS] {label}: :{port} released")
            return True
        time.sleep(0.5)
        if not forced and time.time() > deadline - RELEASE_TIMEOUT / 2:
            forced = True
            for root in roots:
                force(root)

    still = sorted(listening_pids(port))
    print(f"[FAIL] {label}: :{port} still held by {still}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true",
                        help=f"stop only the API on :{BACKEND_PORT}")
    parser.add_argument("--frontend-only", action="store_true",
                        help=f"stop only the UI on :{FRONTEND_PORT}")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be killed, then exit")
    args = parser.parse_args()

    if args.backend_only and args.frontend_only:
        print("[FAIL] --backend-only and --frontend-only are mutually exclusive.")
        return 1

    targets = []
    if not args.frontend_only:
        targets.append((BACKEND_PORT, "backend"))
    if not args.backend_only:
        targets.append((FRONTEND_PORT, "frontend"))

    ok = all([stop_port(port, label, args.dry_run) for port, label in targets])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
