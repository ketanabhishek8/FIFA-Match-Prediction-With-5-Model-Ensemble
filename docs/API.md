# API reference

FastAPI backend in `api/main.py`, serving the model bundle built by
`api/model_cache.py` (see [ARCHITECTURE.md](ARCHITECTURE.md) for how that
bundle gets trained). Three endpoints: list teams, predict a matchup, fetch
evaluation results for the dashboard.

```bash
python -m uvicorn api.main:app --reload
```

Interactive docs (Swagger UI) at `http://localhost:8000/docs` once running.
The frontend never calls this origin directly — it goes through the Vite dev
proxy at `/api/*` (see [FRONTEND.md](FRONTEND.md)).

## Startup behavior

All three endpoints depend on module-level `state`, populated once in
FastAPI's `lifespan()` handler:

1. Loads `config/config.yaml`.
2. Asserts `project.random_state` is one of the hardcoded `SEEDS = [0, 1, 17,
   42, 123]` — raises at startup otherwise, since that value picks which
   seed's report is served as the default.
3. Calls `api.model_cache.load_or_train(cfg, SEEDS)`, which trains the full
   pipeline (classifier ensemble + scoreline model, all 5 seeds) on a cache
   miss, or loads `outputs/models/<hash>.joblib` on a hit.

A cache miss takes several minutes. **The server does not accept traffic
until this finishes** — there is no `/health` endpoint and no partial
readiness; `/teams` is the cheapest call that only succeeds once startup is
done, which is what `start.py` and the frontend's `withRetry()` poll on. A
request that somehow lands before `state` is populated raises `KeyError` (an
unhandled 500), not a clean "still starting" response — see
[LIMITATIONS.md](LIMITATIONS.md#engineering--api).

CORS is wide open (`allow_origins=["*"]`, all methods/headers) — fine for
local development, not configured for any other deployment.

## `GET /teams`

Every (team, year) pair with a squad profile, grouped by team.

**Response** — `200`, `list[{ team: string, years: int[] }]`, sorted by team
name, years ascending:

```json
[
  { "team": "Argentina", "years": [2015, 2016, ..., 2025] },
  { "team": "Brazil",    "years": [2015, 2016, ..., 2025] }
]
```

A team with gaps in `years` means no profile could be built for the missing
edition (e.g. absent from that year's source data) — the frontend's edition
`<select>` in `Predictor.jsx` only offers years present for *both* chosen
teams.

## `POST /predict`

Predicts a single matchup: the 5-model ensemble's winner + per-model votes,
plus (if the scoreline model is enabled) an expected-goals / scoreline /
three-way outcome breakdown.

**Request body:**

```ts
{
  team_a: string,        // required, must be a name /teams returns
  team_b: string,        // required, must differ from team_a
  year?: int,            // FIFA edition to use; omitted = latest shared edition
  neutral?: bool,        // default true
}
```

- `year`, if omitted, defaults to the most recent edition both teams have a
  profile for (`_latest_shared_year`) — the right default for a hypothetical
  future fixture, since there's no player data for a tournament that hasn't
  happened yet.
- `neutral` (default `true`): World Cup finals are played at neutral venues,
  but the model is trained on rows where — across all in-scope competitions,
  draws dropped — 62.8% of *non-neutral* matches are home wins against 49.6%
  of neutral ones. So without correction it would favour whichever team is
  passed as `team_a`. (The README quotes 63.6% / 46.8% for the same split
  restricted to qualifiers; both are measured, on different populations.)
  With `neutral=true`, the fixture is scored in both
  orientations and averaged, making the result order-independent. Set to
  `false` only for a fixture with a genuine home side (the result will then
  carry a real home-advantage bump toward `team_a`).

**Response** — `200`:

```ts
{
  team_a: string,
  team_b: string,
  year: int,                          // the edition actually used
  neutral: bool,
  winner: string,                     // team_a or team_b
  confidence: number,                 // winner's mean soft-vote probability, >= 0.5
  model_votes: { [model: string]: string },       // each of the 5 models' pick
  model_confidence: { [model: string]: number },  // each model's confidence in ITS OWN pick, >= 0.5
  score: {                            // null if score_model.enabled is false,
                                       // or a cached bundle predates the scoreline model
    expected_goals: { team_a: number, team_b: number },
    most_likely: { team_a: int, team_b: int, probability: number },
    top_scorelines: Array<{ team_a: int, team_b: int, probability: number }>,  // top 5
    outcome_probability: { team_a: number, draw: number, team_b: number },
    favours: string,                  // which side the scoreline model favours to win
    agrees_with_ensemble: bool,       // favours == winner
    dixon_coles_rho: number,
  } | null
}
```

Model keys in `model_votes`/`model_confidence` are the config's
`models.classifiers` entries (default: `logistic_regression`,
`random_forest`, `xgboost`, `adaboost`, `knn`).

**`winner`/`confidence` derivation:** these come from the mean of the 5
models' *soft-vote* probabilities, not the ensemble's hard majority vote — so
they can never disagree in direction with each other (unlike combining a
separately-computed hard vote with a soft-vote confidence, which can point
different ways on a narrow majority). This differs from the *offline*
evaluation figures in the README/report, which use the paper's hard
`VotingClassifier`; it only affects this live single-match endpoint.

**`score.agrees_with_ensemble`:** the classifier ensemble has no draw class,
so "agreement" is judged on which side the scoreline model favours to *win*,
ignoring its draw mass. The two models are fit to different targets (who won,
vs. how many goals each side scored) and can legitimately disagree on close
matches — the frontend surfaces this rather than silently picking one.

**Errors:**

| Status | Cause |
|---|---|
| `400` | `team_a == team_b` ("Pick two different teams.") |
| `400` | No FIFA edition has a profile for both teams (only when `year` is omitted — `_latest_shared_year`) |
| `400` | No profile data for one/both teams at the given `year` (`build_single_match_features` → `ValueError`) |

Note the API is the source of truth for the `team_a != team_b` rule
(`api/main.py:184`) even though the frontend also disables its submit button
for that case.

## `GET /evaluation`

Everything the dashboard renders, computed from the cached model bundle. No
request parameters.

**Response** — `200`:

```ts
{
  comparison: Array<{                 // one row per {"Proposed Method", "Baseline Model"}
    "Evaluation Metric": string,
    "Overall Accuracy": number,
    "Accuracy (High-scoring)": number,
    "Accuracy (Low-scoring)": number,
  }>,                                  // single seed (default_seed)

  seed_variance: {                    // ACROSS all 5 seeds
    "Proposed Method": Array<{ seed: int, accuracy: number }>,
    "Baseline Model":  Array<{ seed: int, accuracy: number }>,
  },

  confusion_matrix: {                 // single seed; [[TN, FP], [FN, TP]] over {away win, home win}
    proposed: number[][],
    baseline: number[][],
  },

  pca_cumulative_variance: number[] | null,   // single seed
  per_model_accuracy: { [model: string]: number },  // single seed
  feature_importance: { [feature: string]: number } | null,  // single seed, top 15, PCA-mapped

  roc: { auc: number, points: Array<{ fpr: number, tpr: number }> },  // single seed

  default_seed: int,
  seeds: int[],                       // [0, 1, 17, 42, 123]

  score_model: {                      // null if score_model.enabled is false
    comparison: Array<Record<string, number | string>>,  // rows: fitted model, uncorrected
                                                           // (no Dixon-Coles) variant, league-average floor
    seed_variance: {
      RPS: Array<{ seed: int, [model: string]: number }>,
      "Derived Win Accuracy": Array<{ seed: int, [model: string]: number }>,
    },
    dixon_coles_rho: number,
    dixon_coles_enabled: bool,
    n_train: int,
    n_test: int,
  } | null,

  dataset: {
    n_team_profiles: int,
    n_train_matches: int,
    n_test_matches: int,
    n_raw_features: int,
    n_pca_components: int | null,
  },
}
```

**Scope, panel by panel — read this before adding a new field.** Every block
*except* `seed_variance` (both the classifier's and the scoreline model's) is
computed from **one seed's** held-out test split (`default_seed`). The
5-seed averages live only in `seed_variance`. This is why `Dashboard.jsx`
labels nearly every chart with which scope it's showing — on some seeds the
baseline actually beats the ensemble even though the 5-seed mean favours the
ensemble, and without a label the page would look self-contradictory. If you
add a field to this endpoint, decide which scope it belongs to and label it
in the frontend the same way.

`feature_importance` is `null` whenever PCA is disabled
(`features.use_pca: false`) — there's no PCA loading matrix to map tree
importances back through.

## Shared response conventions

- All numeric probabilities/accuracies are plain floats in `[0, 1]`, not
  percentages — formatting to `%` is a frontend concern (`api.js`/`Dashboard.jsx`).
- Every array-shaped numeric field (`confusion_matrix`, `roc.points`,
  `pca_cumulative_variance`, ...) is already JSON-serializable Python
  (`.tolist()` / explicit `float()`/`int()` casts in `api/main.py`) — no
  numpy types cross the wire.
- Model dictionaries (`model_votes`, `per_model_accuracy`, ...) are keyed by
  the internal config name (`xgboost`, not `"XGBoost"`); the frontend's
  `MODEL_LABEL` in `constants.js` is the display-name mapping.
