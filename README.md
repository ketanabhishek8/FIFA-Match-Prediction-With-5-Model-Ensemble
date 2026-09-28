# FIFA World Cup Match Predictor

A player-aware machine learning framework for predicting FIFA World Cup match outcomes,
built following the methodology in *"From Players to Champions: A Generalizable Machine
Learning Approach for Match Outcome Prediction with Insights from the FIFA World Cup"*
(Al-Bustami & Ghazal, IEEE eIT 2025), with data scope defined in the project proposal.

## Idea

Most existing models predict match winners using only team-level stats (rankings, past
results). National squads are rebuilt every World Cup, so team reputation alone is a weak
signal. This project fuses:

- **Player-level data** — technical, physical, mental, and goalkeeping attributes per
  player, aggregated into **year-specific team profiles** (rosters change every cycle).
- **Team-level data** — historical results, head-to-head record, tournament context.

...into a 5-model ensemble (Logistic Regression, Random Forest, XGBoost, AdaBoost, KNN)
combined via majority voting, benchmarked against a Weighted Win Ratio (WWR) baseline.

## Documentation

This README covers the project's idea, methodology, and results. For code-level reference:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map and data flow, from raw CSVs to the served model.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — environment setup, running tests/pipeline/API, config reference.
- [docs/API.md](docs/API.md) — the FastAPI endpoints (`/teams`, `/predict`, `/evaluation`): request/response shapes and error modes.
- [docs/FRONTEND.md](docs/FRONTEND.md) — the React app's components, API contract, and data flow.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — deploying to Vercel, with the trained model served from Supabase Storage.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — testing gaps, known limitations, and future-improvement ideas.

## Project structure

```
├── config/config.yaml        # all tunable settings in one place
├── data/
│   ├── raw/                  # untouched downloaded data (gitignored)
│   ├── external/             # reference data (rankings, tournament calendars)
│   └── processed/            # cleaned team-year profiles, model-ready match features
├── src/
│   ├── config.py             # config loader + seeding
│   ├── pipeline.py           # classifier task: profiles → features → train/eval per seed
│   ├── score_pipeline.py     # scoreline (Poisson) counterpart to pipeline.py
│   ├── holdout.py            # scores the withheld years (2025) exactly once
│   ├── backtest_wc2026.py    # out-of-sample backtest of the real 2026 knockout bracket
│   ├── make_figures.py       # regenerates everything in outputs/figures/
│   ├── data/
│   │   ├── collect_players.py   # pull player attribute data (per edition)
│   │   ├── collect_matches.py   # pull match history data
│   │   ├── adapters.py          # normalize the EA FC 25/26 schemas onto the FIFA-era columns
│   │   └── clean.py             # squad selection + team-year profile assembly
│   ├── features/
│   │   ├── build_features.py    # Team A/B feature vectors, representation, scaling, PCA
│   │   └── history_features.py  # as-of-date Elo / form / head-to-head / WWR
│   ├── models/
│   │   ├── baseline.py          # Weighted Win Ratio (WWR) baseline — paper §IV
│   │   ├── ensemble.py          # 5-model ensemble + majority voting
│   │   ├── train.py             # CV hyperparameter search
│   │   └── score_model.py       # Poisson + Dixon-Coles scoreline model
│   └── evaluation/
│       └── evaluate.py          # accuracy, high/low-scoring split, McNemar, importances
├── api/
│   ├── main.py               # FastAPI app: /teams, /predict, /evaluation
│   └── model_cache.py        # trains the pipeline once, caches to outputs/models/
├── frontend/                 # React + Vite web app (predictor + results dashboard)
├── tests/                    # unit tests (pytest, run from the repo root)
├── outputs/
│   ├── models/               # cached trained-model bundles (gitignored)
│   └── figures/              # plots for the report
├── docs/                     # architecture, development, API, frontend, limitations
├── report/                   # report.md / report.pdf (Phase 7)
├── notebooks/                # exploratory analysis (currently empty)
├── CONTEXT/                  # the paper, the proposal, the literature survey
├── run_checks.py             # smoke test: env → data → unit tests → pipeline
├── start.py / stop.py        # start/stop backend + frontend together
├── app.py / vercel.json      # Vercel entrypoint and config (docs/DEPLOYMENT.md)
├── CODE_REVIEW.md            # historical line-by-line review (see docs/LIMITATIONS.md)
├── requirements.txt          # runtime dependencies (what the deployment installs)
└── requirements-dev.txt      # + tools for the pipeline, figures, tests, local server
```

## Web app

A small full-stack demo sits on top of the pipeline: pick two teams and a FIFA
edition year for a live ensemble prediction, or browse the evaluation results
as interactive charts.

The quickest way in — one command starts both halves, waits for the backend to
finish training/loading, and opens the browser:

```bash
python start.py
```

`start.py --backend-only` / `--frontend-only` start one half, `--no-browser`
skips opening a window, and anything already listening on `:8000` / `:5173` is
reused rather than started twice. Ctrl+C stops whatever it started.

Because of that reuse, a second `python start.py` finds both ports taken,
opens the browser and exits right away — leaving no process to Ctrl+C. To
shut the servers down in that case:

```bash
python stop.py
```

It kills by port (`--backend-only` / `--frontend-only` for one half,
`--dry-run` to see what it would kill first).

To run the two halves by hand instead:

```bash
# Terminal 1 — backend (trains the pipeline on first run, ~4 min; then caches to
# outputs/models/ and loads from cache instantly on subsequent starts)
python -m uvicorn api.main:app --reload

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open the Vite dev server URL (default `http://localhost:5173`) — it proxies
`/api/*` to the backend on `http://localhost:8000`.

## Setup

Needs **Python ≥ 3.10** (developed on 3.13). The web app additionally needs
Node **^20.19 or ≥ 22.12**, which is what the pinned Vite 8 requires — the
model, tests and figures run on Python alone. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#prerequisites) for what each
dependency is for.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

If pulling datasets from Kaggle, place your API token at `~/.kaggle/kaggle.json`
(see https://www.kaggle.com/docs/api) — `collect_players.py` and `collect_matches.py`
use the `kaggle` CLI for this.

## Phase 2: Data collection (confirmed sources)

Four datasets, confirmed to fit this project's scope — three player sources
(EA renamed the series after FIFA 23, and the original author stopped
publishing after EA FC 24, so 2025+ comes from other authors) and one match
source:

| | Years | Dataset | Slug | File |
|---|---|---|---|---|
| Player-level | 2015–2024 | FIFA Complete Player Dataset (per edition, Stefano Leone) | `stefanoleone992/fifa-{edition}-complete-player-dataset` | `players_{edition}.csv` |
| Player-level | 2025 | EA Sports FC 25 database (nyagami) | `nyagami/ea-sports-fc-25-database-ratings-and-stats` | `male_players.csv` |
| Player-level | 2026 † | EA Sports FC 26 player ratings (justdhia) | `justdhia/ea-sports-fc-26-player-ratings` | `ea_fc26_players.csv` |
| Match-level | 1872– | International football results from 1872–present | `martj42/international-football-results-from-1872-to-2017` | `results.csv` |

† Wired up in `src/data/collect_players.py`'s `MODERN_SOURCES` but not
exercised: `data.years` stops at 2025, and `src/data/adapters.py` notes that
this file's column names have not been checked against a real download.

The 2025 row is not optional detail — the held-out 2025 season that the
headline result is measured on, and the squad ratings behind the 2026 bracket
backtest, both come from the FC 25 source.

For 2015–2024 the player dataset is versioned per FIFA game edition, mapped
roster `year` → `edition = year - 2000` (e.g. 2021 → FIFA 21). Those files
include a `nation_position` field (non-null only for players actually in that
edition's national squad) — `clean.filter_national_team_squad()` uses it to
build accurate squads, rather than filtering on nationality alone (which would
include every player of that nationality, most of whom never get capped).
**The FC-era sources carry no such field**, with consequences described under
[What the 2025 data can't do](#what-the-2025-data-cant-do).

To pull the data:

```bash
# 1. Get a Kaggle API token: kaggle.com/settings -> API -> Create New Token
#    Save it to ~/.kaggle/kaggle.json (chmod 600 on macOS/Linux)

# 2. Run the collectors
python -m src.data.collect_players   # downloads one dataset per configured year
python -m src.data.collect_matches   # downloads the match history dataset
```

Both scripts skip files that are already present in `data/raw/`, so they're
safe to re-run.

## How the phases map to this repo

| Phase | Task | Where it lives |
|---|---|---|
| 1 | Problem definition & literature survey | *(done — see project docs)* |
| 2 | Data collection & cleaning | `src/data/` |
| 3 | Feature engineering (team profiles, scaling, PCA) | `src/features/build_features.py` |
| 4 | Model development (5-model ensemble + majority voting) | `src/models/` |
| 5 | Model evaluation & hyperparameter tuning | `src/models/train.py`, `src/evaluation/` |
| 6 | Testing against baseline | `src/models/baseline.py` + `src/evaluation/evaluate.py` |
| 7 | Report writing | `report/` + `outputs/figures/` |
| 8 | Final presentation / demo | `frontend/` + `api/` (live predictor + results dashboard) |

## Status

- [x] Repo scaffolding
- [x] Data collection (real Kaggle datasets connected: player attributes 2015–2025 — FIFA 15–23,
  EA FC 24 from the same author, EA FC 25 mapped via `src/data/adapters.py` — match history 1872–present)
- [x] Feature engineering (1749 team-year profiles, 64 features -> ~23 PCA components at 95% variance):
  one `<attr>_diff` per squad attribute (means, star-concentration shape, per-position aggregates, EA's six aggregate
  ratings), 6 as-of-date match-history differences (`src/features/history_features.py`), plus neutral-venue and
  squad-size context
- [x] Baseline (WWR) — fit on full World Cup history (finals + qualifiers), test-leakage safe
- [x] Ensemble models — 5-model majority vote, trained end-to-end on real data (1504 usable matches, 1223 after the 2025 holdout is withheld)
- [x] Evaluation — accuracy, high/low-scoring split, challenging cases, validated across 5 random seeds
- [x] Report figures (`outputs/figures/`: accuracy comparison, seed variance, confusion matrices,
  PCA variance, ROC curve, per-model accuracy, feature importance)
- [x] Report writing (Phase 7) — `report/report.md` / `report/report.pdf`
- [x] Web app (Phase 8) — FastAPI backend (`api/`) + React frontend (`frontend/`): live team-vs-team prediction and an interactive results dashboard

**Headline result — the held-out 2025 season, scored once:**

| | accuracy |
|---|---|
| **Proposed ensemble** | **82.2%** |
| Weighted Win Ratio baseline | 75.4% |
| Always predict the home side | 59.4% |

281 matches the model had never been trained, tuned, or compared against
(`python -m src.holdout`). The 6.8-point margin over the baseline is
**statistically significant** — McNemar p=0.0134 — which is the first time that has been
true in this project. ROC-AUC 0.907, and 36 of the 69 matches the baseline gets wrong are
recovered (52.2%).

Two caveats belong next to that number rather than in a footnote. First, **all 281 holdout
matches are qualifiers**. No World Cup finals were played in 2025, so the headline is
measured on qualifying fixtures only — as, largely, is the whole project: finals are 100 of
the 1504 usable matches. Second, **the 2025 squads behind it are proxies, and their
goalkeeping features are not built the way earlier years' were** — see
[What the 2025 data can't do](#what-the-2025-data-cant-do).

Every other number below comes from **development** data (2015–2024, 1504 → 1223 matches
after the holdout is withheld): ensemble **79.0% ± 3.3%** vs. baseline 76.0%, 5-seed average,
ahead on all 5 seeds. Those are the numbers used to choose the model; the holdout is the one
used to report it. See [Why there is a holdout](#why-there-is-a-holdout).
Run `python -m src.pipeline` to reproduce, or `python -m src.make_figures` to
regenerate the figures.

#### How it got there

The ensemble originally scored 75.5% ± 2.5% against the baseline's 74.8% — ahead on only
2 of 5 seeds, and behind on the seed the API serves. The two models were built on disjoint
information: the ensemble knew squads and nothing about results, the baseline knew results
and nothing about squads. They disagreed on about a third of matches, and on the matches the
baseline got wrong the ensemble was right 44% of the time — a perfect router between the two
would have scored 85.9%. The ensemble had simply never been given what the baseline knows.

Three changes, each measured on its own across all 5 seeds:

| | accuracy | margin over baseline | seeds ahead |
|---|---|---|---|
| Starting point | 75.5% ± 2.5% | +0.7 pp | 2 / 5 |
| \+ `neutral_site` feature | 76.3% ± 2.3% | +1.5 pp | 3 / 5 |
| \+ as-of-date history features | 78.1% ± 3.0% | +3.3 pp | 4 / 5 |
| \+ neutral-only row mirroring | 78.5% ± 2.0% | +3.7 pp | 5 / 5 |
| \+ squad shape & position aggregation | 78.6% ± 1.8% | +3.8 pp | 5 / 5 |
| \+ goal-difference-weighted Elo | 78.9% ± 1.9% | +4.2 pp | 5 / 5 |
| \+ `_diff`-only representation | 78.8% ± 1.1 | +4.0 pp | 5 / 5 |
| \+ edition-aware squad join | 80.3% ± 3.2% | +5.3 pp | 5 / 5 |
| \+ EA aggregate ratings (FC 25 adapter) | **79.0% ± 3.3%** † | **+3.0 pp** | **5 / 5** |

† measured on development only, after the 2025 holdout was withheld — not comparable row-to-row with the
lines above it, which were measured on the full set. On the same development data the previous
configuration scores 78.6%, so the adapter is worth about +0.4 pp.

- **History features** (`src/features/history_features.py`) give the model Elo, the baseline's
  own weighted win ratio, matches played, head-to-head record and recent form. Every value is
  computed *as of the match date* from matches strictly earlier than it, so leak-freedom is
  structural rather than a property of the split — same-day fixtures cannot see each other
  either. They are built from the full international record, not just World Cup matches,
  because Elo and form need volume.
- **`neutral_site`** matters on its own. Among **qualifiers with draws dropped**, the home side
  wins 63.6% of them; on neutral ground it is 46.8%. (Over all in-scope competitions rather than
  qualifiers alone, the same split is roughly 63% / 49% — the figure quoted in `api/main.py`.)
  Without the flag the model blends the two
  and applies a phantom home advantage to every neutral-venue fixture — which is every match at
  the tournament itself.
- **Mirroring** (appending a side-swapped copy of a training row with the label flipped) only
  pays off restricted to neutral fixtures. Mirroring everything *cost* ~2 points, since
  asserting that the away side of a qualifier had home advantage is simply false. Restricted to
  rows where the ordering really is arbitrary it adds 0.4 points and cuts seed-to-seed spread
  from 3.0 to 2.0 — which is what turns "ahead on 4 of 5 seeds" into "ahead on all 5".
- **Squad shape and position aggregation** came out of a single bad prediction: Brazil 78% over
  a Norway side that won 2-1. The squad *mean* was the one statistic making it look lopsided —
  Brazil 84.4 to Norway's 76.6 — while by top-3 the gap was 1.7 and by best player Norway was
  ahead (Haaland 91 to 90). The mean measures depth, which wins leagues; a knockout tie is
  often decided by the best two or three on the pitch. Profiles now also carry `overall_max`,
  `overall_top3`, `overall_spread` and per-position aggregates. This also surfaced a real bug:
  the proxy squad for unlicensed federations was "top 23 by `overall`", and since EA rates
  keepers below outfielders, **58% of those squads contained no goalkeeper** (Algeria 2017's
  best keeper ranks 32nd in his own nation). Selection now fills a positional quota. Net effect
  on accuracy is only +0.1 pp — inside the noise — but AUC, seed spread and significance all
  improve, and Brazil–Norway drops from 78% to 72%.
- **Elo weighting**, following the World Football Elo convention, split in two when measured
  separately — and only half of it survived:

  | | accuracy | margin | vs flat K=20 |
  |---|---|---|---|
  | flat K=20 | 78.60% ± 1.75 | +3.8 pp | reference |
  | competition-weighted (WC 60 … friendly 20) | 78.26% ± 1.27 | +3.5 pp | −0.33 pp |
  | goal-difference multiplier | **78.93% ± 1.89** | **+4.2 pp** | **+0.33 pp** |
  | both | 78.13% ± 1.99 | +3.3 pp | −0.47 pp |

  Competition weighting is the textbook choice and it made things slightly worse on 4 of 5
  seeds. Tying K to the tournament makes every rating move faster, so Elo becomes reactive to
  recent results — which is what `form_win` and `form_gd` already measure. What the ensemble
  wants from Elo is the stable long-run reputation a flat K gives it. The goal-difference
  multiplier is kept: margin of victory is real evidence no other feature carries per match.
  It is left in the code behind `features.history_elo_competition_weighted` for re-measuring.

- **`_diff`-only representation and an edition-aware squad join** came out of a full audit.
  `<attr>_a`, `<attr>_b` and `<attr>_diff` are collinear by construction, and `h2h_winrate`'s
  three columns are correlated at r = 1.0000 exactly — 163 features carrying far less than 163
  features' worth of information. Dropping to differences alone (163 → 56 on the dataset as it
  stood then; 188 → 64 today, after the FC 25 columns were mapped) held accuracy and
  cut seed spread from 1.9 to 1.1. The bigger win was the join: FIFA edition *Y* ships around
  September of *Y−1*, but squads were being looked up by the match's **calendar year**, and 68%
  of matches in scope are played September–December — so most fixtures were reading ratings a
  median of 12 months old while a fresher edition already existed. Joining on the edition
  current at the match date moves 55% of rows onto newer squads and is worth **+1.5 pp**, the
  largest single gain of any change here. It does widen seed spread (1.1 → 3.2), which is the
  one cost.
- **Squad-size reliability did not work**, in either form. Profiles are built from whoever EA
  rated, so a "team" can be 2 players: 33% of match rows have a side under 5, and 48% under 11.
  Exposing `squad_size_min` as a feature moved nothing (−0.06 pp), and *filtering* thin training
  rows was actively harmful — −0.9 pp at a threshold of 11 and −3.1 pp at 18. The lost volume
  costs more than the added noise. The feature is kept (it is one column and genuinely
  describes the row); `features.min_squad_size` ships at 0.

Per-seed McNemar does **not** reach p<0.05 on any development seed (p = 0.188, 0.272, 0.230,
0.391, 0.617; n=245 per test set) — a 3-point gap on 245 rows is simply underpowered. What
changed is that the margin is positive on every seed rather than two. The significance is on
the holdout, where the larger +6.8-point margin over 281 matches reaches p=0.0134.

#### Why there is a holdout

Roughly fifteen configurations were compared against the same five random splits over the
course of this work, each kept or discarded on how it scored there. That is model selection on
a test set, and its effect compounds quietly. It showed up directly: across the same stretch
the random-split figure climbed 78.9% → 80.3%, while an untuned temporal split went
79.7% → 79.5%. The last round of "gains" had not replicated.

So **2025 is now withheld entirely** (`evaluation.holdout_years`). `build_match_dataset()`
drops it, which means it never reaches feature selection, hyperparameter tuning, the seed
variance charts, the figures, or the served model. The only thing that reads it is
`python -m src.holdout`, which scores it once and says so.

```bash
python -m src.holdout
```

The cost is real: 281 fewer training matches, which is why the 2026 bracket drops from 86.7%
to 83.3%. The benefit is that **82.2%** is a number no decision was made against.

One caveat kept honest: the holdout was locked *after* most configuration choices had already
been made with 2025 visible, so it is clean for everything from the FC 25 adapter onward, not
for the whole history of the project. That it came in *above* the development estimate rather
than below is the reassuring direction.

#### What the 2025 data can't do

The holdout season is the one year built from a different publisher's file
(`nyagami`'s EA FC 25 database, normalized by `src/data/adapters.py`), and two
differences carry straight into the headline number:

- **Every 2025 squad is a proxy.** The FC-era sources carry no national-team
  call-up tags at all, so there is no `nation_position` to filter on and every
  nation falls through `clean.filter_national_team_squad()`'s top-N-by-`overall`
  path. That fallback was already load-bearing for unlicensed federations in
  earlier years — this is a difference of degree, not of kind — but it means the
  season the model is *reported* on is built entirely from proxy squads rather
  than real rosters.
- **Goalkeeping features are not comparable across the 2024/2025 boundary.** The
  FIFA-era files gave every outfielder a low `goalkeeping_*` value; the FC-era
  sources populate those attributes only for actual keepers. A 2025 profile's
  goalkeeping means therefore describe only its keepers, where a 2019 profile's
  describe the whole squad. Arguably the newer behaviour is more correct, but the
  two are not built the same way, so the model meets a distribution shift on
  those columns exactly at the train/holdout line.

Neither is a reason to discount the result — but "82.2% on unseen 2025" means
"on proxy squads with rebuilt goalkeeping features", and that is worth stating
next to the number. `src/data/adapters.py`'s module docstring is the canonical
list of what the mapping cannot recover.

#### Validity checks

Two things that would invalidate the headline number were tested directly rather than assumed:

- **Is the random split hiding that this trains on the future?** No. Splitting *temporally*
  instead — train on everything up to 2024-10-15, predict the 295 matches after it — gives
  **79.7% ± 0.7% against the baseline's 77.0%**. Accuracy does not drop, so the random-split
  figure is not inflated by hindsight, and the forecasting framing is defensible.
- **Is the baseline handicapped?** Slightly, and it was worth checking: the ensemble's history
  features are built from all 49,518 international matches while the WWR baseline is fit on the
  9,839 World Cup ones — a 5× information edge. Giving the baseline the full record moves it
  from 74.8% to 74.9%, so the like-for-like margin is **+4.0 pp rather than +4.2 pp** at the
  point that was measured. Real, but small enough that it does not change the conclusion.

### Held-out test: the 2026 World Cup knockout bracket

`results.csv` contains the 2026 World Cup, but `data.years` deliberately stops at **2025**, so
no 2026 match reaches the ensemble or the WWR baseline. Team strength comes from EA FC 25
squad ratings, published before the tournament began on 2026-06-11, and the history features
are cut at that same kickoff date — so a 2026 result cannot inform a 2026 prediction. That
makes the knockout bracket a genuine out-of-sample test rather than a cross-validation fold:

```bash
python -m src.backtest_wc2026
```

| Round | Ties | Ensemble | Baseline |
|---|---|---|---|
| Round of 32 | 16 | 88% | 50% |
| Round of 16 | 8 | 88% | 88% |
| Quarter-finals | 4 | 100% | 100% |
| Semi-finals | 2 | 0% | 50% |
| **Overall** | **30** | **83.3%** | **66.7%** |

83.3% out of sample sits above the development figure of 79.0% ± 3.3%, which is the main evidence
that the model generalises rather than fitting its own test splits — though *above* is itself a
warning sign rather than a triumph. The goal-difference Elo change was worth +0.33 pp in
cross-validation and appears to be worth +10 pp here; that gap is three matches out of thirty,
so most of it is luck, and the cross-validated figure is the one to quote. Two more caveats:
with n=30 the 95% interval is ±13.3%, so the 17-point margin over the baseline is **not**
statistically significant; and 4 of the 30 ties were decided on penalties, which the model has
no way to represent — it predicts a 90-minute winner and is scored against the shootout result.

The model gets both semi-finals wrong, picking France over Spain and England over Argentina. Note this
bracket is now predicted by a model trained without 2025 matches at all, since they are held out — 281
fewer training matches, which costs it roughly one tie.

Neither the third-place playoff nor the final has a recorded score in this dataset, so both are
genuine forward predictions: **England** to finish third (62%), and **Spain** to beat Argentina
in the final (52%) — barely above a coin flip, which is the correct amount of conviction for a
World Cup final between those two.

**Beyond overall accuracy**, `python -m src.pipeline` also reports (console + `report.attrs`):
- Per-model accuracy for each of the 5 base classifiers, before the majority vote
- Precision / recall / F1 / ROC-AUC (accuracy alone hides the home-win class imbalance)
- McNemar's test on paired baseline-vs-ensemble predictions — the ~3.0-point development
  accuracy margin reaches p<0.05 on **none** of the 5 seeds (p = 0.188–0.617; honest finding,
  not swept under the rug — see the report's Discussion section). With 245 test rows per seed
  the test is underpowered for a gap this size; the stronger evidence on development data is
  that the margin is positive on all 5 seeds, and the holdout's larger margin *is* significant
- Ensemble accuracy broken down by vote agreement (3/5, 4/5, 5/5 of the base models agreeing) —
  pooled over the 5 seeds, a unanimous vote is right 83.9% of the time against 56.2% for a
  split one
- Feature importance (Random Forest + XGBoost, mapped back from PCA-component space to
  the original 64 engineered features) — an approximation, since PCA components have no
  direct real-world meaning

## Reference

Al-Bustami, A. & Ghazal, Z. "From Players to Champions: A Generalizable Machine Learning
Approach for Match Outcome Prediction with Insights from the FIFA World Cup." 2025 IEEE
International Conference on Electro Information Technology (eIT), pp. 574–578.
DOI: 10.1109/EIT64391.2025.11103598
