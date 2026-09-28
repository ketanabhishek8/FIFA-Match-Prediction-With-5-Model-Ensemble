# Development guide

Setting up the project, running it day-to-day, and the config knobs you're
likely to touch. For what the code does and how it's organized, see
[ARCHITECTURE.md](ARCHITECTURE.md); for the frontend, [FRONTEND.md](FRONTEND.md).

## Prerequisites

| | Version | Why |
|---|---|---|
| Python | **≥ 3.10** (developed on 3.13) | PEP 604 `X \| None` annotations throughout, including Pydantic request models in `api/main.py` that resolve them at runtime. |
| Node | **^20.19 or ≥ 22.12** | Only needed for the frontend — it's what the pinned Vite 8 declares in its `engines` field. |
| Kaggle account | — | The player and match datasets are pulled with the `kaggle` CLI (see below). |

Node is not needed for the model itself: the pipeline, tests, figures and the
API all run on Python alone. `start.py` shells out to `npm`, so it needs Node
unless you pass `--backend-only`.

### What the Python dependencies are for

`requirements.txt` holds only what the API needs at runtime — it is what the
Vercel deployment installs ([DEPLOYMENT.md](DEPLOYMENT.md)) — and
`requirements-dev.txt` adds everything else on top of it. In short: **pandas/numpy** for the data
frames the whole pipeline passes around; **scikit-learn** for the five
classifiers, `GridSearchCV`, scaling and PCA, plus **xgboost** for the one
model it doesn't provide; **scipy** for the distributions behind the scoreline
model and McNemar's test, and **joblib** for the cached model bundles (both
imported directly, not just as scikit-learn transitives); **matplotlib** and
**seaborn** for `outputs/figures/`; **fastapi**/**uvicorn** to serve the model
and **httpx** for the API test; **kaggle** for data collection; **pytest** for
the suite; **jupyter**/**ipykernel**/**tqdm** for ad-hoc work.

The frontend's dependencies are in `frontend/package.json` — React 19, Vite 8,
and `recharts` for the dashboard charts, with `oxlint` for linting. See
[FRONTEND.md](FRONTEND.md).

## Environment setup

```bash
python -m venv venv             # python3.10+ -- see Prerequisites
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

Data comes from Kaggle. Place your API token at `~/.kaggle/kaggle.json`
(`kaggle.com/settings → API → Create New Token`, `chmod 600` on macOS/Linux),
then:

```bash
python -m src.data.collect_players   # one dataset per configured year -> data/raw/
python -m src.data.collect_matches   # match history -> data/raw/results.csv
```

Both collectors skip files already present in `data/raw/`, so they're safe to
re-run. If you already have the CSVs (e.g. copied from another checkout), just
place them under `data/raw/` with the expected names
(`players_<year>.csv`, `results.csv`, `shootouts.csv`) and skip collection
entirely.

## Running things

| Command | What it does |
|---|---|
| `python run_checks.py` | Fast smoke test: env → data presence → unit tests (excl. `test_api.py`) → single-seed pipeline. Run this after any change. |
| `python run_checks.py --full` | Everything above plus all 5 seeds and `test_api.py` (which boots the API and trains on a cache miss — slow). |
| `pytest` | Unit tests only, run from the repo root (no `pytest.ini`/`pyproject.toml` — tests import `src.*`, which only resolves relative to the repo root). |
| `python -m src.pipeline` | Full classifier pipeline: build profiles → features → train/eval ensemble vs. baseline, over the 5 hardcoded seeds `[0, 1, 17, 42, 123]`. |
| `python -m src.score_pipeline` | Same, for the scoreline (Poisson) model. |
| `python -m src.holdout` | Scores `evaluation.holdout_years` (2025) exactly once, on a model trained on all development data. See the README's "Why there is a holdout" section before running this repeatedly — it stops meaning anything if you tune against it. |
| `python -m src.backtest_wc2026` | Backtests the real 2026 World Cup knockout bracket against a model that has never seen a 2026 match. |
| `python -m src.make_figures` | Regenerates every PNG in `outputs/figures/`. |
| `python start.py` | Starts backend (uvicorn) + frontend (Vite) together, waits for the backend to finish training/loading, opens a browser. `--backend-only` / `--frontend-only` / `--no-browser` flags available. |
| `python stop.py` | Kills whatever `start.py` left running, by port (`:8000` / `:5173`), since a second `start.py` invocation that finds both ports taken exits immediately with nothing to Ctrl+C. `--dry-run` to preview. |

`python run_checks.py` is the closest thing to a pre-commit gate this project
has — run it before committing anything that touches `src/`, `api/`, or
`config/config.yaml`.

## Model cache

`api/main.py` trains the pipeline once and caches the result to
`outputs/models/<hash>.joblib` (`api/model_cache.py`). The hash is derived
from the `data`, `features`, `baseline`, `models`, `evaluation`, and
`score_model` sections of `config.yaml` plus the seed list and
`project.random_state` — changing any of those invalidates the cache and
triggers a retrain on the next server start. Changing `project.name` or other
config outside those sections does *not* invalidate it.

There is no cache eviction — every distinct config you've run leaves a
`.joblib` file behind in `outputs/models/`. Delete stale ones by hand if disk
space matters. These files are `joblib` pickles: only load ones this repo
produced, never one from an untrusted source.

`project.random_state` must be one of the hardcoded `SEEDS` list in
`api/main.py:20` (currently `[0, 1, 17, 42, 123]`) — the API startup check
raises if it isn't, since that value picks which seed's report is served as the
default. If you change it, keep it in sync with the other three copies:
`src/make_figures.py:15` (also a named `SEEDS`), `src/pipeline.py:362` and
`run_checks.py:144` (inline literals, so grepping `SEEDS` won't find them).
There's no single source of truth for it yet — see
[LIMITATIONS.md](LIMITATIONS.md#engineering--api).

## Config reference (`config/config.yaml`)

The file is heavily commented in place with the *measured* effect of each
knob (accuracy deltas, seed variance) — read it alongside this table rather
than instead of it.

| Section | Key knobs |
|---|---|
| `project` | `random_state` — default seed; must be in `api/main.py`'s `SEEDS` if you're running the API. |
| `data` | `years` — which FIFA editions / match-history years are in scope (2026 is deliberately excluded so the bracket backtest stays out-of-sample). `include_qualifiers` — widen training beyond WC finals. |
| `features` | `representation` (`all` vs `diff`), `edition_aware_join`, `min_squad_size`, `history_features` + its Elo/form knobs, `mirror_training_rows`, `use_pca` + `pca_variance_threshold`, `scaler`. |
| `baseline` | `min_head_to_head_matches`, `wwr_m` — the WWR baseline's two parameters. |
| `models` | `test_size`, `cv_folds`, `classifiers` — the list of the 5 model keys to include (see the caveat below before removing one). |
| `score_model` | `enabled`, `max_goals`, `alpha`, `dixon_coles` + `rho_bounds`/`rho_grid_points`. |
| `evaluation` | `high_scoring_goal_threshold`, `holdout_years` — years withheld from all development work; set to `[]` only if you understand you're giving up the "never optimised against" guarantee (see README). |

**Caveat on `models.classifiers`:** dropping `random_forest` or `xgboost` from
this list will raise a `KeyError` in `src/pipeline.py:314`'s feature-importance
step, which hardcodes those two names. Either keep both in the list or patch
that step first. Similarly, shrinking the list below 5
will desync the frontend's hardcoded model-count assumptions — see
[FRONTEND.md](FRONTEND.md#watch-out-for).

## Tests

| File | Covers |
|---|---|
| `test_baseline.py` | `WeightedWinRatioBaseline` — WWR formula, head-to-head tiebreaking. |
| `test_build_features.py` | `_a`/`_b`/`_diff` join, draw dropping, label correctness, `build_single_match_features` error cases. |
| `test_clean_profiles.py` | `filter_national_team_squad`'s licensed-vs-proxy branches, squad quota backfill. |
| `test_history_features.py` | `HistoryState` (Elo update math, WWR-in-features parity with the baseline, as-of-date snapshotting, same-day exclusion). |
| `test_score_model.py` | Dixon-Coles tau mass-preservation, `PoissonScoreModel` fit/predict, RPS/log-loss sanity. |
| `test_evaluate.py` | McNemar's test, feature-importance-through-PCA mapping. |
| `test_api.py` | End-to-end: boots the FastAPI app and hits `/teams`, `/predict`, `/evaluation`. Slow (trains on a cache miss) — excluded from `run_checks.py`'s default (non-`--full`) run. |

Known gaps (see [LIMITATIONS.md](LIMITATIONS.md#testing) for the current
list): no determinism test asserting `run_seed()` is bit-for-bit reproducible
at a fixed seed, and `test_api.py` currently accepts a 400 response as a pass
on `/predict` regardless of whether the pair should have succeeded.

## Known issues / where to look before extending the model

[LIMITATIONS.md](LIMITATIONS.md) is the current, up-to-date list of testing
gaps, known limitations (data/scope, methodology, engineering), and
future-improvement ideas — read it before making changes near
`src/pipeline.py`, `src/models/`, or `api/main.py`.

`CODE_REVIEW.md` at the repo root is an earlier, dated line-by-line review
(severity-tagged `[H]`/`[M]`/`[L]`) written against a specific commit; several
of its items are already fixed (noted as such in `LIMITATIONS.md`) and it's
kept as historical record rather than a live checklist.
