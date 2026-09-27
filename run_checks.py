"""Smoke-test the project end to end: environment -> data -> unit tests ->
pipeline. Run this after changing anything to confirm the whole chain still
works, and to see where it breaks if it doesn't.

    python run_checks.py           # fast: unit tests + single-seed pipeline
    python run_checks.py --full    # everything: 5-seed pipeline + API test

Every stage prints [PASS] / [FAIL] / [SKIP] and the script exits non-zero if
any stage failed, so it also works as a pre-commit gate.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Imported by the modelling pipeline. Names are import names, not pip names.
CORE_DEPS = ["numpy", "pandas", "sklearn", "xgboost", "yaml", "scipy", "joblib"]
FIGURE_DEPS = ["matplotlib", "seaborn"]
API_DEPS = ["fastapi", "httpx"]

results: list[tuple[str, str]] = []


def stage(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def check_environment() -> bool:
    """Confirm the interpreter actually has the project's dependencies.

    This is the stage that catches "wrong interpreter" -- e.g. a bare system
    Python instead of the project venv, which fails much more confusingly later.
    """
    stage("1. Environment")
    print(f"interpreter : {sys.executable}")
    print(f"version     : {sys.version.split()[0]}")
    print(f"project root: {ROOT}\n")

    missing = []
    for mod in CORE_DEPS + FIGURE_DEPS + API_DEPS:
        try:
            importlib.import_module(mod)
            print(f"  ok      {mod}")
        except ImportError:
            print(f"  MISSING {mod}")
            missing.append(mod)

    if missing:
        record("environment", "FAIL", f"{len(missing)} missing: {', '.join(missing)}")
        print(
            "\n  Install them into a virtual environment:\n"
            "      python -m venv venv\n"
            "      venv\\Scripts\\activate          (Windows)\n"
            "      source venv/bin/activate       (macOS/Linux)\n"
            "      pip install -r requirements.txt\n"
            "  then re-run this script with that venv's python."
        )
        return False

    record("environment", "PASS", f"all {len(CORE_DEPS + FIGURE_DEPS + API_DEPS)} imports ok")
    return True


def check_data() -> bool:
    """The raw Kaggle CSVs are gitignored, so a fresh clone has none of them."""
    stage("2. Raw data")
    from src.config import load_config, resolve_path

    cfg = load_config()
    raw = resolve_path(cfg["data"]["raw_dir"])

    missing = []
    results_csv = raw / "results.csv"
    if results_csv.exists():
        print(f"  ok      results.csv ({results_csv.stat().st_size // 1024} KB)")
    else:
        print("  MISSING results.csv")
        missing.append("results.csv")

    for year in cfg["data"]["years"]:
        path = raw / f"players_{year}.csv"
        if path.exists():
            print(f"  ok      players_{year}.csv ({path.stat().st_size // 1024} KB)")
        else:
            print(f"  MISSING players_{year}.csv")
            missing.append(f"players_{year}.csv")

    if missing:
        record("raw data", "FAIL", f"{len(missing)} file(s) missing")
        print(
            "\n  Download them (needs a Kaggle token at ~/.kaggle/kaggle.json):\n"
            "      python -m src.data.collect_players\n"
            "      python -m src.data.collect_matches"
        )
        return False

    record("raw data", "PASS", f"results.csv + {len(cfg['data']['years'])} player files")
    return True


def check_unit_tests(full: bool) -> bool:
    """Fast unit tests only by default.

    tests/test_api.py is excluded unless --full: it boots the API, which trains
    the entire pipeline on a cache miss (minutes, not seconds).
    """
    stage("3. Unit tests")
    targets = ["tests/test_baseline.py", "tests/test_evaluate.py"]
    if full:
        targets.append("tests/test_api.py")
        print("  (including test_api.py -- trains the pipeline on a cache miss)\n")
    else:
        print("  (skipping test_api.py -- use --full to include it)\n")

    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q"],
        cwd=ROOT,
    )
    elapsed = time.time() - started

    if proc.returncode == 0:
        record("unit tests", "PASS", f"{len(targets)} file(s) in {elapsed:.1f}s")
        return True
    record("unit tests", "FAIL", f"pytest exited {proc.returncode}")
    return False


def check_pipeline(full: bool) -> bool:
    """The real end-to-end check: build profiles -> features -> train -> evaluate."""
    stage("4. Pipeline")
    seeds = [0, 1, 17, 42, 123] if full else [42]
    print(f"  running seed(s): {seeds}"
          + ("" if full else "  (single seed -- use --full for all 5)") + "\n")

    from sklearn.model_selection import train_test_split

    from src.config import load_config
    from src.pipeline import build_match_dataset, build_team_profiles, run_seed

    started = time.time()
    cfg = load_config()
    profiles = build_team_profiles(cfg)
    feat, wc_all_history, _history_state = build_match_dataset(cfg, profiles)
    print(f"\n  team-year profiles : {len(profiles)}")
    print(f"  usable matches     : {len(feat)}")
    print(f"  WC history rows    : {len(wc_all_history)}\n")

    reports = {s: run_seed(cfg, feat, wc_all_history, s) for s in seeds}
    elapsed = time.time() - started

    record("pipeline", "PASS", f"{len(seeds)} seed(s) in {elapsed:.0f}s")
    _verify_no_imputation_leak(cfg, feat, reports[seeds[0]], seeds[0], train_test_split)

    if len(seeds) > 1:
        import pandas as pd

        # .to_dict() rather than pd.concat -- see the same note in src/pipeline.py:
        # reports carry numpy arrays in .attrs, which pandas >= 3 compares during
        # concat and chokes on.
        overall = pd.DataFrame({s: r["Overall Accuracy"].to_dict() for s, r in reports.items()}).T
        print("\n  Overall accuracy across seeds:")
        print("  " + overall.to_string().replace("\n", "\n  "))
        print("\n  " + overall.agg(["mean", "std"]).to_string().replace("\n", "\n  "))
    return True


def _verify_no_imputation_leak(cfg, feat, report, seed, train_test_split) -> None:
    """Confirm `train_mean` really is the training-split mean, not the mean of
    the whole frame -- i.e. that the imputation leak stays fixed."""
    import numpy as np

    attrs = report.attrs
    feature_cols = attrs["feature_cols"]
    X, y = feat[feature_cols], feat["label"]

    idx_train, _ = train_test_split(
        feat.index, test_size=cfg["models"]["test_size"], random_state=seed, stratify=y,
    )
    expected = X.loc[idx_train].mean()
    full_frame = X.mean()

    if not np.allclose(attrs["train_mean"], expected, equal_nan=True):
        record("no imputation leak", "FAIL", "train_mean != training-split mean")
        return

    if np.allclose(expected, full_frame, equal_nan=True):
        record("no imputation leak", "PASS",
               "train_mean == training-split mean (no NaNs present, so the "
               "split makes no difference here)")
    else:
        record("no imputation leak", "PASS",
               "train_mean == training-split mean, and differs from the "
               "full-frame mean as expected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="run all 5 seeds and include the slow API test")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    if not check_environment():
        return _summary()
    if not check_data():
        return _summary()

    tests_ok = check_unit_tests(args.full)

    try:
        check_pipeline(args.full)
    except Exception:
        record("pipeline", "FAIL", "raised an exception")
        traceback.print_exc()

    if not tests_ok:
        print("\n  note: unit tests failed above -- pipeline output may still "
              "look fine while a component is subtly broken.")
    return _summary()


def _summary() -> int:
    stage("Summary")
    width = max((len(name) for name, _ in results), default=0)
    for name, status in results:
        print(f"  {name.ljust(width)}  [{status}]")

    failed = [n for n, s in results if s == "FAIL"]
    if failed:
        print(f"\n  {len(failed)} stage(s) FAILED: {', '.join(failed)}")
        return 1
    print("\n  All stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
