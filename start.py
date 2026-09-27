"""Start the whole app with one command: FastAPI backend + Vite frontend.

    python start.py                 # start both, open the browser when ready
    python start.py --no-browser    # start both, don't open a browser
    python start.py --backend-only  # just the API (http://localhost:8000)
    python start.py --frontend-only # just the UI (expects a backend already up)

The backend trains (or loads its cache from outputs/models/) on the first
request, so the first startup can take a few minutes; this script waits for it
to answer before opening the browser. Ctrl+C stops both processes.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
VENV_PYTHON = ROOT / "venv" / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"
# Backend readiness probe. /teams is the cheapest endpoint that only answers
# once the model bundle is trained or loaded from cache.
READY_URL = f"{BACKEND_URL}/teams"
BACKEND_TIMEOUT = 900  # seconds; a cold (untrained) start is slow


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def port_in_use(port: int) -> bool:
    """True if something is already listening on localhost:port.

    Checks every address family localhost resolves to: Vite binds ::1 while
    uvicorn binds 127.0.0.1, so probing only one of them misses the other.
    """
    for family, socktype, proto, _, addr in socket.getaddrinfo(
        "localhost", port, type=socket.SOCK_STREAM
    ):
        with socket.socket(family, socktype, proto) as sock:
            sock.settimeout(1)
            if sock.connect_ex(addr) == 0:
                return True
    return False


def npm_command() -> str:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        fail("npm not found on PATH - install Node.js to run the frontend.")
    return npm


def backend_python() -> str:
    """The interpreter to run uvicorn with.

    Plain `python start.py` on a system interpreter without the project's deps
    would spawn a backend that dies immediately on "No module named uvicorn",
    and the only symptom here would be the unhelpful "backend exited before it
    became ready" - so check up front, and fall back to the repo's venv when
    the interpreter running this script can't import uvicorn itself.
    """
    if importlib.util.find_spec("uvicorn") is not None:
        return sys.executable
    if VENV_PYTHON.exists() and subprocess.run(
        [str(VENV_PYTHON), "-c", "import uvicorn"], capture_output=True
    ).returncode == 0:
        print(f"[..] backend: no uvicorn for {sys.executable},")
        print(f"     using {VENV_PYTHON} instead")
        return str(VENV_PYTHON)
    print("[FAIL] uvicorn is not installed for this interpreter:")
    print(f"       {sys.executable}")
    print(f"       ...and there is no usable venv at {VENV_PYTHON}")
    print("       Create the venv and install the deps, then re-run:")
    print("         python -m venv venv")
    print("         pip install -r requirements.txt")
    sys.exit(1)


def start_backend() -> subprocess.Popen:
    python = backend_python()
    print(f"[..] backend: uvicorn api.main:app on :{BACKEND_PORT}")
    return subprocess.Popen(
        [python, "-m", "uvicorn", "api.main:app",
         "--port", str(BACKEND_PORT)],
        cwd=ROOT,
    )


def start_frontend() -> subprocess.Popen:
    npm = npm_command()
    if not (FRONTEND / "node_modules").exists():
        print("[..] frontend: node_modules missing, running npm install")
        if subprocess.run([npm, "install"], cwd=FRONTEND).returncode != 0:
            fail("npm install failed.")
    print(f"[..] frontend: vite dev server on :{FRONTEND_PORT}")
    # --strictPort so a clash fails loudly rather than silently moving to
    # another port and leaving the URL we print (and open) wrong.
    return subprocess.Popen(
        [npm, "run", "dev", "--", "--port", str(FRONTEND_PORT), "--strictPort"],
        cwd=FRONTEND,
    )


def wait_for_backend(proc: subprocess.Popen, timeout: int = BACKEND_TIMEOUT) -> bool:
    """Poll the API until it answers. Returns False on timeout or early exit."""
    print("[..] waiting for the backend (first run trains the models, be patient)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print("[FAIL] backend exited before it became ready.")
            return False
        try:
            with urllib.request.urlopen(READY_URL, timeout=5) as resp:
                if resp.status == 200:
                    print("[PASS] backend ready")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    print(f"[FAIL] backend not ready after {timeout}s.")
    return False


def stop(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            if sys.platform == "win32":
                # `npm run dev` is a wrapper: terminating it leaves the real
                # vite process orphaned and still holding the port, so kill
                # the whole tree.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open a browser window")
    parser.add_argument("--backend-only", action="store_true",
                        help="start only the FastAPI backend")
    parser.add_argument("--frontend-only", action="store_true",
                        help="start only the Vite frontend")
    args = parser.parse_args()

    if args.backend_only and args.frontend_only:
        fail("--backend-only and --frontend-only are mutually exclusive.")

    procs: list[subprocess.Popen] = []
    try:
        if not args.frontend_only:
            if port_in_use(BACKEND_PORT):
                # Reuse it rather than starting a second uvicorn that can only
                # fail to bind.
                print(f"[..] backend: something already on :{BACKEND_PORT}, "
                      "reusing it")
            else:
                backend = start_backend()
                procs.append(backend)
                if not wait_for_backend(backend):
                    stop(procs)
                    return 1

        if not args.backend_only:
            if port_in_use(FRONTEND_PORT):
                print(f"[..] frontend: something already on :{FRONTEND_PORT}, "
                      "reusing it")
            else:
                procs.append(start_frontend())
            url = FRONTEND_URL
        else:
            url = f"{BACKEND_URL}/docs"

        if not args.no_browser:
            # Give Vite a moment to bind its port before pointing a browser at it.
            time.sleep(3)
            print(f"[..] opening {url}")
            webbrowser.open(url)

        if not procs:
            print("\nEverything was already running - nothing to manage.\n")
            return 0

        print("\nRunning. Press Ctrl+C to stop.\n")
        while True:
            for proc in procs:
                if proc.poll() is not None:
                    print("[FAIL] a process exited; shutting the rest down.")
                    stop(procs)
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[..] stopping")
        return 0
    finally:
        stop(procs)


if __name__ == "__main__":
    sys.exit(main())
