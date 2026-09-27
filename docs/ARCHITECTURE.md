# Architecture

How the pieces fit together: raw data in, a trained ensemble + scoreline model
out, served by an API and shown in a web UI. This is a reference for the code
layout and data flow — for *why* specific modeling choices were made (feature
representation, mirroring, the holdout), see the root [README](../README.md),
which documents the experimentation trail in detail.

## Data flow, end to end

```
data/raw/*.csv (gitignored)
        │
        ▼
src/data/collect_players.py ──► load_player_data()   (concat players_<year>.csv, tag with `year`)
src/data/collect_matches.py ──► load_match_data()     (results.csv, parse dates, tag with `year`)
src/data/adapters.py         (normalizes the 2025/2026 FC-branded sources onto
                               the same column names collect_players.py expects)
        │
        ▼
src/data/clean.py
   filter_national_team_squad()   -- real national-team rosters where EA tagged
                                      them (`nation_position`), else a proxy squad
                                      (top players by `overall`, filled by a
                                      positional quota so it always has a keeper)
   clean_player_data()            -- drop sparse columns / rows missing identifiers
   build_all_team_year_profiles() -- one row per (team, year): squad means,
                                      overall_max/top3/spread, per-position aggregates
        │
        ▼  team-year profiles (src/pipeline.build_team_profiles)
        │
        ▼
src/features/build_features.py
   build_match_features()   -- join profiles onto each match as `<attr>_a` /
                                `<attr>_b` / `<attr>_diff`, using the FIFA
                                edition current on the match date
                                (edition_aware_join), attach `neutral_site`,
                                drop draws (classifier) or keep them (scoreline)
src/features/history_features.py
   build_history_features() -- Elo, WWR, matches played, head-to-head, recent
                                form, computed AS OF each match date from a
                                HistoryState advanced day-by-day over the FULL
                                international record (not just World Cup matches)
        │
        ▼  feature frame (src/pipeline.build_match_dataset)
        │      `evaluation.holdout_years` (2025) is dropped here, so it never
        │      reaches tuning, seed-variance reporting, or the served model
        ▼
src/pipeline.prepare_split()   -- per split: thin-squad filter, impute from
                                   TRAIN mean, mirror neutral-venue rows
                                   (side-swap + flipped label), scale, PCA
        │
        ├──► src/models/train.py + src/models/ensemble.py
        │       5-fold CV per model type -> best_estimator_ -> hard-vote
        │       VotingClassifier (Logistic Regression, Random Forest,
        │       XGBoost, AdaBoost, KNN)
        │
        ├──► src/models/baseline.py
        │       WeightedWinRatioBaseline, fit on WC history minus the test rows
        │
        └──► src/models/score_model.py (via src/score_pipeline.py, same split
                machinery, draws kept)
                PoissonScoreModel: two PoissonRegressors (home/away goals) +
                Dixon-Coles low-score correction -> a scoreline distribution
        │
        ▼
src/evaluation/evaluate.py   -- accuracy by scoring bucket, McNemar, ROC-AUC,
                                 per-model accuracy, PCA-mapped feature
                                 importance, RPS/log-loss/derived-win-accuracy
                                 for the scoreline model
        │
        ▼
api/model_cache.py   -- trains once per (data/features/baseline/models/
                         evaluation/score_model config + seeds), caches the
                         whole bundle (profiles, per-seed reports, history
                         state) to outputs/models/<hash>.joblib
        │
        ▼
api/main.py (FastAPI)  ──►  frontend/ (React + Vite)
```

Two standalone entry points read the *withheld* years directly, bypassing the
cache-and-serve path above: `src/holdout.py` (scores 2025 once) and
`src/backtest_wc2026.py` (backtests the real 2026 knockout bracket, training
years fixed at ≤2025 with history features cut at kickoff).

## Module map

| Module | Responsibility |
|---|---|
| `src/config.py` | Loads `config/config.yaml`; `set_seed()` seeds `random` + `numpy`. Every other module takes `cfg: dict` rather than reading the file itself. |
| `src/data/collect_players.py` | Kaggle download + column-group constants (`ATTACKING_COLUMNS`, `GOALKEEPING_COLUMNS`, ...) for the per-edition player datasets; `_edition_for_year()` maps roster year → FIFA edition number. |
| `src/data/collect_matches.py` | Kaggle download for `results.csv`; `filter_world_cup_matches()` restricts to WC finals (+ optionally qualifiers, per `data.include_qualifiers`). `add_net_score()` is **dead code** — `net_score` survives only in `get_feature_columns()`'s exclude set. |
| `src/data/adapters.py` | Column-name mapping for the post-FIFA-23 sources (EA Sports FC 25/26), which ship under different schemas from different authors. Documents exactly what it *cannot* recover (e.g. `nation_position`). |
| `src/data/clean.py` | `filter_national_team_squad()` (licensed-vs-proxy squad selection), `build_all_team_year_profiles()` (the player→team aggregation step: squad means + shape statistics + position-restricted aggregates). `drop_redundant_features()` is **dead code** — the paper's non-redundancy step, defined but never called; PCA stands in for it. |
| `src/features/build_features.py` | Turns (match, team-year profiles) into a model-ready row: `edition_for_match_date()`, `_attach_team_profiles()` (the `_a`/`_b`/`_diff` join), `build_match_features()` (training), `build_single_match_features()` (one-row inference), `mirror_feature_frame()`, `get_feature_columns()` (representation: `all` vs `diff`), `scale_features()`, `apply_pca()`. |
| `src/features/history_features.py` | `HistoryState`: a running per-team Elo/WWR/form/head-to-head snapshot, advanced match-by-match. `build_history_features()` produces as-of-date features for a target frame; `build_history_state()` just advances the state (used by the 2026 backtest, which needs a state frozen the day before kickoff). |
| `src/models/baseline.py` | `WeightedWinRatioBaseline` — the paper's WWR baseline: head-to-head record above a match-count threshold, else a credibility-weighted win ratio. |
| `src/models/ensemble.py` | `PARAM_GRIDS` (GridSearchCV grids per model type) and `build_base_estimators()` / `build_majority_vote_ensemble()`. |
| `src/models/train.py` | `tune_all_models()` — per-model 5-fold `GridSearchCV`, seeded explicitly so multi-seed runs vary model randomness, not just the split. Also contains `run_training_pipeline()`, **dead code**: a second training path with no scaling and no PCA that nothing calls. `pipeline.run_seed()` is the real one. |
| `src/models/score_model.py` | `PoissonScoreModel` (two Poisson regressions + Dixon-Coles correction → scoreline grid), `ConstantPoissonBaseline` (no-skill floor), and the shared grid/outcome utilities (`scoreline_grid`, `outcome_proba_from_grid`, `outcome_index`) that `evaluate.py` and the API both import. |
| `src/evaluation/evaluate.py` | All scoring: classification metrics, McNemar's test, PCA-mapped feature importance, and the RPS/log-loss/derived-win-accuracy family for the scoreline model. |
| `src/pipeline.py` | Orchestrates the classifier task: `build_team_profiles()`, `build_match_dataset()` (join + history features + holdout split), `prepare_split()` (the single leakage-sensitive impute/mirror/scale/PCA path shared with the scoreline task), `run_seed()` (one full train/eval cycle), `run()` (multi-seed driver, `__main__`). |
| `src/score_pipeline.py` | The scoreline counterpart to `pipeline.py` — same data build and `prepare_split()`, different target (goals, not win/loss), draws kept. |
| `src/holdout.py` | Scores `evaluation.holdout_years` exactly once, training on all development data with no inner split. The only caller that reads the withheld years through the normal pipeline machinery. |
| `src/backtest_wc2026.py` | Predicts the real 2026 World Cup knockout bracket with a `HistoryState` frozen before kickoff — a genuine out-of-sample test, not a CV fold. |
| `src/make_figures.py` | Regenerates every PNG in `outputs/figures/` from a 5-seed pipeline run. |
| `api/model_cache.py` | `load_or_train()` — hashes the training-relevant config sections + seeds into a cache key, and reuses `outputs/models/<hash>.joblib` across server restarts. |
| `api/main.py` | FastAPI app. Trains (or loads) the model bundle in `lifespan()`; see [API.md](API.md) for endpoint details. |

## Design points that show up in more than one place

- **As-of-date history features are the reason the random train/test split is
  safe.** `HistoryState` only ever sees matches strictly before the fixture
  being scored (same-day matches are snapshotted before any of that day's
  results are applied), so leak-freedom is structural rather than a property
  of any particular split.
- **`prepare_split()` is the one leakage-sensitive code path**, shared by
  `pipeline.run_seed()` and `score_pipeline.run_seed_scores()`. Imputation,
  mirroring, scaling and PCA are all fit on the training rows of *that split*
  only; the two tasks differ solely in which target they carry through the
  mirrored rows (`mirror_targets()`, called separately by each caller).
- **Squad profiles and match features are built once, reused across seeds.**
  `pipeline.run()` builds `profiles` and `feat` outside the seed loop —
  everything below that point (train/test split, model fit) is what varies
  per seed.
- **The scoreline model shares almost everything with the classifier except
  the target and the draw filter.** Same config, same `build_match_dataset()`
  call (with `drop_draws=False`), same `prepare_split()`. It trains and caches
  separately because a draws-inclusive frame needs its own scaler/PCA fit.
- **The API's model bundle is a cache, not a service.** `api/model_cache.py`
  trains synchronously in FastAPI's `lifespan()` handler on a cache miss —
  the first request-serving moment is gated on the whole pipeline finishing.
  `start.py` waits on `/teams` for exactly this reason.
