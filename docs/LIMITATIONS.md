# Testing, limitations & future improvements

What the test suite actually covers, what's known to be wrong or fragile
today, and what would move the project forward. Written against the current
codebase (not a specific past commit) — see [CODE_REVIEW.md](../CODE_REVIEW.md)
at the repo root for the dated line-by-line review this doc supersedes as the
place to check current status; several items there are already fixed and
noted as such here.

## Testing

The full test/command reference (what each file covers, how to run things) is
in [DEVELOPMENT.md](DEVELOPMENT.md#tests) — this section covers the gaps and
the reasoning behind them rather than repeating that table.

**What's covered:** the WWR baseline formula and tie-breaking, the
`_a`/`_b`/`_diff` feature join and label correctness, the national-squad
filter's licensed/unlicensed branches, the Elo/history-feature math (including
as-of-date snapshotting and same-day exclusion), the Dixon-Coles scoreline
model, McNemar's test and the PCA-to-original-feature importance mapping, and
an end-to-end API smoke test.

**What isn't:**

- **No determinism test.** Nothing asserts that `run_seed(cfg, feat, hist, 42)`
  gives bit-identical output across two runs. Cheap to add, and it's the kind
  of thing that silently breaks when a dependency changes its RNG behavior.
- **`test_api.py` treats a 400 as a pass regardless of whether the pair should
  have succeeded** (`tests/test_api.py`). If `/predict` regressed to erroring
  on every request, this test would stay green. It should derive a pair with
  known-overlapping profile years from `/teams` and require 200.
- **No `pytest.ini` / `pyproject.toml`.** Tests only resolve `src.*` imports
  because pytest happens to be run from the repo root; there's no config
  pinning that down.
- **No frontend tests.** The React app (`frontend/`) has no test files at all
  — coverage is manual (see [FRONTEND.md](FRONTEND.md)).
- `test_api.py` is also the slowest test by far (trains the full pipeline on
  a cache miss), which is why `run_checks.py` excludes it by default and only
  runs it under `--full`.

## Limitations

### Data & scope

- **Player quality is a video-game rating, not a ground-truth skill measure.**
  Every player-level feature ultimately comes from EA's FIFA/FC attribute
  ratings. That's the whole premise of the paper this project follows, but
  it means the model inherits whatever bias EA's rating process has (e.g.
  reputation effects, uneven scouting coverage of smaller federations).
- **Many squads are a proxy, not the real roster — and in 2025+, all of them
  are.** `filter_national_team_squad` falls back to "top N by rating with a
  positional quota" whenever EA didn't tag national-team call-ups. For
  2015–2024 that affects unlicensed federations only. The FC-era sources
  (2025+) carry no `nation_position` field at all (`src/data/adapters.py`), so
  **every** 2025 squad is a proxy — including the entire holdout season the
  headline result is measured on.
- **Goalkeeping features are not comparable across the 2024/2025 boundary.**
  FIFA-era files gave every outfielder a low `goalkeeping_*` value; the FC-era
  sources populate those attributes only for actual keepers. A 2025 profile's
  goalkeeping means therefore describe only its keepers, where earlier years'
  describe the whole squad — a distribution shift on those columns landing
  exactly at the train/holdout line. `src/data/adapters.py`'s docstring is the
  canonical list of what the schema mapping cannot recover.
- **The headline number is measured on qualifiers, not finals.** The 2025
  holdout is 281 qualifying matches — no World Cup finals were played that
  year. Only 100 of the 1504 usable development matches are finals at all, so
  a model nominally "for" World Cup prediction is trained and validated
  overwhelmingly on non-tournament fixtures.
- **No representation for penalty shootouts.** The 2026 bracket backtest
  scores 4 of 30 ties on their shootout result while the model only ever
  predicts a 90-minute winner — those are unwinnable by construction, not
  model error.
- **History features (Elo, form, WWR-in-features) are built from the full
  international match record**, not just World Cup matches, because Elo and
  form need volume to be meaningful. That's a deliberate choice (documented
  in the README's validity checks), but it does mean "history" mixes
  friendlies, qualifiers, and finals rather than being tournament-specific.

### Methodology

- **The WWR baseline's tie-break still favors `team_a`** (`>=` in
  `src/models/baseline.py`), and since home wins are the majority class this
  very slightly flatters the baseline on ties. Small effect, but not zero.
- **PCA feature importance is an approximation.** Because PCA components have
  no direct real-world meaning, mapping Random Forest / XGBoost importances
  back to the original 64 engineered features (`src/pipeline.py`) is a
  best-effort attribution, not an exact one.
- **Development-era model selection happened before the holdout was locked.**
  Per the README's "Why there is a holdout" section, most configuration
  choices were made with 2025 visible before the holdout was formalized — so
  the holdout is clean for changes from the FC 25 adapter onward, not for the
  project's full history. The holdout scoring *above* the development
  estimate is reassuring but doesn't erase that order of operations.
- **Per-seed statistical power is thin.** McNemar's test doesn't reach p<0.05
  on any individual development seed (n≈245 per test set) — a real, positive
  margin on all 5 seeds is the actual evidence there; the single-seed p-values
  aren't informative on their own.
- **Seed-to-seed variance is real** (79.0% ± 3.3% on development). Any single
  run's accuracy number should be read as a sample from that spread, not as
  the model's "true" accuracy.

### Engineering & API

- **Four hardcoded copies of the seed list** — `api/main.py:20` and
  `src/make_figures.py:15` (as a named `SEEDS` constant), `src/pipeline.py:362`
  and `run_checks.py:144` (as inline literals, so grepping `SEEDS` finds only
  half of them). All four must be kept in sync by hand; there's no single
  source of truth yet.
- **`models.classifiers` can't safely drop `random_forest` or `xgboost`.**
  `src/pipeline.py`'s feature-importance step reads those two keys directly
  and raises `KeyError` if either is removed from config.
- **The model count (5) is hardcoded downstream** in `tests/test_api.py`
  (`len(model_votes) == 5`) and `frontend/src/components/Dashboard.jsx`.
  Changing `models.classifiers`'s length desyncs both.
- **No `/health` or readiness endpoint.** The lifespan handler trains the
  full pipeline (~4 min on a cache miss) before the server accepts traffic;
  a request that somehow lands before `state` is populated gets a bare
  `KeyError` → 500 rather than a friendly "still training" response, and the
  frontend has no signal to show a real progress state.
- **CORS is wide open** (`allow_origins=["*"]`, all methods/headers). Fine for
  local development; would need tightening to the actual frontend origin
  before this is hosted anywhere reachable.
- **No cache eviction in `outputs/models/`.** Every distinct config change
  leaves another multi-MB `.joblib` bundle behind; nothing prunes stale ones.
  These are unpickled with `joblib.load` on server start — only ever put
  trusted, repo-produced bundles in that directory.
- **`scipy` and `joblib` are imported directly** (`src/evaluation/evaluate.py`,
  `src/models/score_model.py`, `src/models/train.py`, `api/model_cache.py`)
  but aren't listed in `requirements.txt` — they currently resolve only as
  transitive scikit-learn dependencies, which is one upstream version bump
  away from an `ImportError`.
- **`drop_redundant_features` (`src/data/clean.py`) is dead code** — defined,
  never called. PCA is what actually manages feature collinearity today;
  either wire this in ahead of PCA or remove it so it stops implying an
  unapplied step from the reference paper's methodology.

## Future improvements

**Data & modeling**

- Extend the holdout as new FIFA/FC editions and World Cup cycles release,
  building a rolling multi-year out-of-sample track record instead of the
  current single 2025 holdout and one-off 2026 bracket backtest.
- Model penalty-shootout outcomes for knockout ties, or at minimum report
  shootout-decided ties separately rather than scoring them as ordinary
  90-minute misses.
- Re-measure competition-weighted Elo (`features.history_elo_competition_weighted`,
  currently off by default) periodically as more tournament data accumulates
  — it lost to flat-K on the data available when it was tested, but that
  could change with volume.
- Explore a graph/sequence model over player-level data, per the literature
  survey gap noted in `requirements.txt` (commented-out `torch` /
  `torch-geometric` dependencies) — a natural extension beyond the current
  squad-aggregate features.
- Extend the Dixon-Coles scoreline model's calibration for full bracket
  simulation (e.g. Monte Carlo tournament simulation from single-match
  probabilities), rather than only single-match expected scorelines.

**Engineering**

- Move the seed list into `config.yaml` as the single source of truth, read by
  `api/main.py`, `src/pipeline.py`, `src/make_figures.py`, and `run_checks.py`
  alike.
- Add a `/health` endpoint and have startup-incomplete requests return 503
  with a clear message instead of an unguarded `KeyError` → 500.
- Add the missing determinism test and fix `test_api.py`'s 400-as-pass gap.
- Add a minimal `pytest.ini`/`pyproject.toml` so `pytest` works from any
  working directory.
- Add cache eviction (or a `--clear-cache` flag) for `outputs/models/`.
- Pin `scipy` and `joblib` explicitly in `requirements.txt`.
- Tighten CORS to the actual frontend origin before any non-local deployment.
