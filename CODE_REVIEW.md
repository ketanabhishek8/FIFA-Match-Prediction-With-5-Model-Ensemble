# Code Review Checklist

> **Historical snapshot.** This review is pinned to the commit below; several
> items here are already fixed. **Every line number below is as of that commit
> and no longer resolves** — e.g. §1's feature-importance hardcode, cited as
> `src/pipeline.py:146`, now lives at `src/pipeline.py:314`. Treat the file
> references as pointers to a symbol, not a location, and don't follow the
> `path:line` links.
>
> For the current, maintained list of testing gaps, limitations, and future
> improvements, see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

Review of the repo as of `1aa0ad6` plus the uncommitted changes to
`api/model_cache.py`, `src/models/train.py`, `src/pipeline.py`, and the new
`run_checks.py`.

Each item is a checkbox so it can be worked through and ticked off. Severity:
**[H]** = affects reported results or breaks under a supported config,
**[M]** = real defect but bounded, **[L]** = hygiene / docs / perf.
Items marked **(verify)** are things I could not confirm without running the
pipeline (no `venv` in the working tree) — treat them as "check this", not
"this is broken".

---

## 0. Already fixed in the working tree — keep

- [x] Imputation now uses the training-split mean only ([src/pipeline.py:65](src/pipeline.py:65)), and the `train_mean` stored in `report.attrs` is that same value, so `/predict` imputes consistently with training.
- [x] `tune_all_models` takes an explicit `seed` ([src/models/train.py:24](src/models/train.py:24)) so multi-seed runs vary model randomness, not just the split.
- [x] `project.random_state` is part of the API cache key ([api/model_cache.py:25](api/model_cache.py:25)), so changing the default served seed invalidates the bundle.
- [x] `run_checks.py` asserts the imputation fix stays fixed (`_verify_no_imputation_leak`).

---

## 1. Correctness — breaks under a supported config **[H]**

- [ ] **PCA feature importance hard-codes two classifiers.** [src/pipeline.py:146](src/pipeline.py:146) reads `tuned["random_forest"]` and `tuned["xgboost"]` directly. `config.yaml`'s `models.classifiers` list *is* honoured by `tune_all_models` ([src/models/train.py:39](src/models/train.py:39)), so dropping either model from that list raises `KeyError` at the end of every seed. Guard on membership, or average whatever tree models are present.
- [ ] **The 5-model count is hard-coded downstream.** [tests/test_api.py:31](tests/test_api.py:31) asserts `len(model_votes) == 5` and [frontend/src/components/Dashboard.jsx:9](frontend/src/components/Dashboard.jsx:9) hard-codes a 5-name `MODELS` array. Shrinking `models.classifiers` fails the test and mislabels the dashboard.
- [ ] **`state` is only populated in the lifespan handler.** Any request that lands before/without startup ([api/main.py:54](api/main.py:54), :63, :102) raises `KeyError` → 500. Return 503 with a "model still training" message instead.
- [ ] **`/predict` does not reject `team_a == team_b`.** [api/main.py:62](api/main.py:62) will build a self-match row where every `_diff` feature is 0 and return a meaningless ~50% winner. The frontend disables the button ([Predictor.jsx:78](frontend/src/components/Predictor.jsx:78)), but the API is the contract — return 400.

## 2. Methodology — affects what the reported numbers mean **[H]**

- [ ] **The train/test split is random, not temporal.** [src/pipeline.py:61](src/pipeline.py:61) splits across all years at once, so the model routinely trains on 2022 matches to predict 2018 ones. For a forecasting claim that is optimistic. Either switch to a year-based / group-by-tournament split, or state in the report that the claim is "given contemporaneous data", not "forecasting".
- [ ] **Home/away ordering is an artifact for World Cup finals.** Most WC finals are at neutral venues, and `results.csv`'s `neutral` flag is currently only *excluded* from features ([build_features.py:68](src/features/build_features.py:68)), never used. Which team lands in `_a` vs `_b` is therefore near-arbitrary, yet the label is "home wins". Consider mirroring every training row (swap `_a`/`_b`, flip the label) so the model cannot learn the ordering artifact, and check class balance before/after.
- [ ] **`_a` / `_b` / `_diff` are exactly collinear** by construction ([build_features.py:31](src/features/build_features.py:31)) — `diff = a − b` adds no information, only 1.5× the dimensionality on ~118 rows. PCA partly absorbs it. Measure accuracy with `_diff` only vs. all three and keep the smaller set if it ties.
- [ ] **Sample size vs. claim strength.** With `test_size: 0.2` on ~118 usable matches, each seed's test set is ~24 rows — one flipped match is ~4pp. McNemar's `b + c` will be single digits, so the p-value is near-meaningless. Prefer repeated stratified CV over a single hold-out per seed, and report confidence intervals rather than point accuracies.
- [ ] **The WWR baseline has a built-in home bias.** [baseline.py:97](src/models/baseline.py:97) breaks WWR ties with `>=`, i.e. in favour of `team_a`, and [pipeline.py:108](src/pipeline.py:108) scores that as a home-win prediction. Since home wins are the majority class, this quietly flatters the baseline. Break ties explicitly (majority class, or seeded random) and say which.
- [ ] **The baseline sees matches that postdate the test set.** It is fit on all WC history minus the exact test rows ([pipeline.py:99](src/pipeline.py:99)). That is consistent with the ML side (also non-temporal), but state it in the report — a reader will assume the baseline is historical-only.
- [ ] **Squad-mean aggregation dilutes goalkeeping attributes.** [clean.py:115](src/data/clean.py:115) means every numeric attribute over the whole ~23-player squad, so a team's `goalkeeping_*` values are mostly outfield players' near-zero GK ratings. The docstring already flags position-aware aggregation as a Phase-3 idea — either do it, or note the limitation wherever GK features appear in the importance chart.
- [ ] **Verify the FIFA-edition ↔ match-year alignment.** `players_2018.csv` is FIFA 18, released Sept 2017 — the 2017/18 season squad — joined against calendar-2018 matches ([pipeline.py:48](src/pipeline.py:48)). Defensible, but it is a one-season offset from "the squad that played the tournament". Document the choice, or shift the join by a year and compare. **(verify)**
- [ ] **The paper's non-redundancy principle is never applied.** `drop_redundant_features` ([clean.py:82](src/data/clean.py:82)) is defined and never called. Either call it (before PCA) or delete it and note in the module docstring that PCA stands in for step 3.

## 3. Robustness / dead code **[M]**

- [ ] **`run_training_pipeline` is a second, divergent training path.** [src/models/train.py:57](src/models/train.py:57) splits, tunes and fits with no scaling and no PCA, using `cfg["project"]["random_state"]` rather than a passed seed. Nothing calls it; `src/pipeline.run_seed` is the real implementation. Delete it, or reduce it to a thin wrapper, before someone trusts its output.
- [ ] **`add_net_score` is dead** ([collect_matches.py:94](src/data/collect_matches.py:94)) — `net_score` appears only in `get_feature_columns`' exclude set. Remove one or the other.
- [ ] **`config.data.match_dataset_slug` is unused.** [collect_matches.py:31](src/data/collect_matches.py:31) hard-codes `MATCH_DATASET_SLUG` and the downloader uses the constant, so editing the config value silently does nothing. Read it from `cfg`.
- [ ] **The baseline's test-row exclusion is a row-wise `apply`** over the full WC history ([pipeline.py:99](src/pipeline.py:99)) — O(history × seeds). A merge or `MultiIndex.isin` on `(date, home_team, away_team)` is faster and less fragile about the `date` dtype. Confirm the key types match on both sides after `parse_dates`. **(verify)**
- [ ] **`build_all_team_year_profiles` rescans the whole player frame per (team, year)** ([clean.py:110](src/data/clean.py:110)). A single `groupby(["nationality_name", "year"])[numeric_cols].mean()` plus `size()` gives the identical result in one pass.
- [ ] **Check the dtype of the team-year profile frame.** `build_team_year_profile` returns a `Series` that starts numeric and then has a string `team` assigned into it ([clean.py:116](src/data/clean.py:116)), so each row is `object`-dtype and `pd.DataFrame(rows)` may produce `object` columns. If so, the `_diff` subtraction and `StandardScaler` are working on boxed Python floats. Add an explicit `astype(float)` on the numeric columns and assert it. **(verify)**
- [ ] **`agreement_vs_accuracy` recomputes the majority vote** from base predictions ([evaluate.py:79](src/evaluation/evaluate.py:79)) instead of using the ensemble's own `ml_pred`. Identical today for 5 hard votes; silently wrong if `voting` becomes `"soft"` or the model count changes. Pass `ml_pred` in, or assert the two agree.
- [ ] **`full_comparison_report` mixes numpy arrays with an index-carrying Series.** `total_goals` keeps `feat`'s index ([pipeline.py:110](src/pipeline.py:110)) while `y_true`/`y_pred` are positional arrays ([evaluate.py:121](src/evaluation/evaluate.py:121)). It works today; pass `total_goals.to_numpy()` so alignment can never bite. **(verify)**
- [ ] **`WeightedWinRatioBaseline.predict` on an empty frame** returns an empty `DataFrame`, not a `Series` ([baseline.py:102](src/models/baseline.py:102)) — a latent `AttributeError` downstream. Short-circuit on `matches_df.empty`.
- [ ] **`_wwr` divides by `v + m`** ([baseline.py:82](src/models/baseline.py:82)); with `wwr_m: 0` and an unseen team that is 0/0. Guard, or document that `wwr_m > 0` is required.
- [ ] **No cache eviction in `outputs/models/`** ([model_cache.py:32](api/model_cache.py:32)) — every config change leaves another multi-MB joblib bundle behind. Also note in the docstring that these are pickles: only load bundles this repo produced.

## 4. API / frontend **[M/L]**

- [ ] **`SEEDS` is duplicated in three places** — [api/main.py:19](api/main.py:19), [src/pipeline.py:189](src/pipeline.py:189), and `run_checks.py`. Move the list into `config.yaml` and have all three read it; the startup check in `api/main.py` then becomes a real invariant instead of a reminder to hand-sync.
- [ ] **CORS is wide open** (`allow_origins=["*"]`, all methods, [api/main.py:41](api/main.py:41)). Fine on localhost; tighten to the Vite origin before this is hosted anywhere.
- [ ] **`Dashboard.jsx`'s `MODELS` array duplicates `MODEL_LABEL`'s values** ([Dashboard.jsx:9](frontend/src/components/Dashboard.jsx:9)), directly contradicting the "single source of truth" comment in [constants.js:1](frontend/src/constants.js:1). Use `Object.values(MODEL_LABEL)`.
- [ ] **Two different standard deviations are reported for the same quantity.** `stdev` in the dashboard is the population SD (÷n, [Dashboard.jsx:15](frontend/src/components/Dashboard.jsx:15)); the pipeline's seed summary uses pandas `.std()`, the sample SD (÷n−1, [pipeline.py:185](src/pipeline.py:185)). With 5 seeds that is a ~12% difference in the printed ±. Pick one.
- [ ] **Cold start blocks with no signal.** The lifespan handler trains the whole pipeline before the server accepts traffic ([api/main.py:33](api/main.py:33)), so the frontend just hangs on its first `getTeams()`. A `/health` endpoint (or a startup log the README points at) lets the UI say "training, ~4 min".

## 5. Tests **[M]**

- [ ] **`test_api.py` accepts a 400 as a pass** ([tests/test_api.py:28](tests/test_api.py:28)). If `/predict` regressed to erroring on every pair, the suite stays green. Derive a pair with overlapping profile years from `/teams` and require 200.
- [ ] **No tests for `src/features/build_features.py`** — the module that decides labels. Cover: draws are dropped, `label == 1` really means home win, `_diff == _a − _b`, and `build_single_match_features` raises on an unknown team/year.
- [ ] **No tests for `clean.filter_national_team_squad`** — the licensed/unlicensed fallback is described as "load-bearing, not cosmetic" and is the single biggest driver of how many matches survive. Cover both branches on a small synthetic frame.
- [ ] **No determinism test.** Assert `run_seed(cfg, feat, hist, 42)` twice gives identical accuracy — cheap insurance for the seeding work just done in `train.py`.
- [ ] **No `pytest.ini` / `pyproject.toml`.** Tests only import `src.*` because pytest happens to run from the repo root. Add a minimal config with `testpaths` so `pytest` works from anywhere.

## 6. Docs, config, packaging **[L]**

- [ ] **`requirements.txt` is missing `scipy` and `joblib`**, both imported directly ([evaluate.py:12](src/evaluation/evaluate.py:12), [model_cache.py:11](api/model_cache.py:11), [train.py:17](src/models/train.py:17)). They resolve today only as transitive scikit-learn deps — one upstream change from an ImportError.
- [ ] **`collect_matches.py`'s docstring contradicts the config.** It says the project restricts to `tournament == "FIFA World Cup"`, "finals only, not qualifiers" ([collect_matches.py:14](src/data/collect_matches.py:14)), while `config.yaml` ships `include_qualifiers: true`.
- [ ] **Stale "TODO (Phase 2)" / "skeleton" blocks** describing work already done: [collect_matches.py:18](src/data/collect_matches.py:18), [collect_players.py:46](src/data/collect_players.py:46), [train.py:11](src/models/train.py:11) ("runnable skeleton… depends on Phase 2-3"), [ensemble.py:14](src/models/ensemble.py:14) ("they'll need tuning once real feature data is available"), [baseline.py:22](src/models/baseline.py:22) ("fully implemented (unlike the other models here)").
- [ ] **Two config values still marked "tune later"**: `baseline.wwr_m: 10` ("tune empirically in Phase 5") and `evaluation.high_scoring_goal_threshold: 3` ("tune during EDA"). Either run the sweep and record the justification, or drop the comments so they aren't read as open work.
- [ ] **`train.py`'s `__main__` raises a Phase-2-3 message** ([train.py:82](src/models/train.py:82)) — point it at `python -m src.pipeline` instead.
- [ ] **`run_checks.py` is untracked.** Commit it (it is genuinely useful as a pre-commit gate) and add it to the README's Commands section next to `pytest`.
- [ ] **The README documents the `SEEDS`/`project.random_state` hand-sync as the design.** Update it if the `SEEDS`-in-config item lands.

---

## Suggested order

1. **§1** — small, mechanical, and each one is a crash under a config the repo advertises as supported.
2. **§5** test gaps for `build_features` / `filter_national_team_squad` — do these *before* §2, so the methodology changes have a safety net.
3. **§2** — the split strategy and the home/away artifact are the two that move the headline number; decide them together, then re-run `python run_checks.py --full`.
4. **§3 / §4 / §6** cleanup, batched into one commit.
