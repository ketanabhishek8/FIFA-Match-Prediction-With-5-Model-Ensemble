# Frontend

A React + Vite single-page app with two panels: a live team-vs-team predictor
and an evaluation-results dashboard. Both talk to the FastAPI backend in
`api/` (see [ARCHITECTURE.md](ARCHITECTURE.md)) through a Vite dev-server
proxy, never directly.

This is the app's documentation; `frontend/README.md` is a short pointer here
plus the npm commands.

## Layout

```
frontend/src/
├── main.jsx           entry point, mounts <App/>
├── App.jsx            header + renders <Predictor/> and <Dashboard/>
├── api.js             fetch wrapper: withRetry(), getTeams(), predict(), getEvaluation()
├── constants.js        MODEL_LABEL (canonical model names), humanizeFeature()
├── teamSearch.js       ranked fuzzy search for the team comboboxes
├── components/
│   ├── Predictor.jsx    the "pick two teams" form + prediction result
│   ├── Dashboard.jsx    the evaluation charts (recharts)
│   └── TeamCombobox.jsx type-to-search team picker used twice in Predictor
├── App.css / index.css  styling (CSS custom properties, no framework)
└── assets/              static images (hero art, icons)
```

## Data flow

`api.js` is the only module that calls `fetch`. Its `BASE = '/api'` requests
are rewritten by the Vite dev proxy (`vite.config.js`) to
`http://localhost:8000` — the app never hardcodes the backend origin, so the
same code runs against any port the backend happens to be on locally.

```
Predictor.jsx / Dashboard.jsx
        │  useEffect(() => withRetry(getTeams | getEvaluation, ...))
        ▼
api.js: request()  ──►  fetch('/api/...')  ──►  vite proxy  ──►  uvicorn :8000
```

### `withRetry` and cold starts

The backend trains the whole pipeline in FastAPI's `lifespan()` handler on a
cache miss (`api/model_cache.py`) — several minutes on the very first run.
Before that finishes, requests either get connection-refused or come back as
a 5xx with no JSON body, which through the Vite proxy is indistinguishable
from the server being broken. `api.js` treats `500/502/503/504` **without** a
`detail` field as *transient* ("still starting") rather than an error, and
`withRetry()` polls every 3s (up to 100 attempts) until the backend answers or
returns a real `detail`-bearing error (e.g. an unknown team, a 400). Both
`Predictor` and `Dashboard` drive their initial load through `withRetry` and
show a "waiting for the backend…" hint (`starting` state) while it's polling.

### `/predict` request/response shape

See [API.md](API.md) for the full request/response shape. In short:

```js
predict(team_a, team_b, year, neutral = true)
// -> { winner, confidence, model_votes, model_confidence, score: {...} | null }
```

`neutral` (default `true`) tells the backend to score the fixture in both
orientations and average — the model is trained on rows where a "home" team
is listed even for neutral-venue matches, so without this the winner would
depend on which team happened to be passed as `team_a`. The UI only exposes
switching it off as an explicit toggle in `Predictor.jsx`, with a note about
the home-side bias that introduces.

`score` is `null` whenever `score_model.enabled` is false in `config.yaml` or
the cached bundle predates the scoreline model — `Predictor.jsx` renders the
scoreline block conditionally (`result.score && ...`) rather than assuming
it's always present. Same pattern in `Dashboard.jsx` for `data.score_model`.

## Components

### `Predictor.jsx`

- Fetches `/teams` once, defaults to `Brazil vs Germany` if both are present
  (else the first two teams returned).
- `years` is the intersection of the two teams' available FIFA editions
  (`a.years ∩ b.years`); the edition `<select>` is disabled until both teams
  have at least one shared year, and auto-selects the most recent one
  whenever the candidate set changes.
- Renders three result blocks once a prediction comes back: the winner +
  per-model consensus bar, the scoreline panel (expected goals, top-5
  scorelines, three-way outcome bar — see `agrees_with_ensemble` in
  `api/main.py`'s `_score_prediction()` for why the two models can legitimately
  disagree), and the per-model vote list.

### `Dashboard.jsx`

Fetches `/evaluation` once and renders it as a fixed sequence of panels: a
stat band (accuracy delta, dataset size, PCA components), an accuracy-by-scope
bar chart, confusion matrices, a 5-seed stability line chart, PCA cumulative
variance, per-model accuracy, an ROC curve, the scoreline comparison table,
and top feature importances.

**Scope labels matter here.** Most panels come from a single seed's held-out
test split (`data.default_seed`); only the stat tiles and the stability chart
average across all 5 seeds. Every chart title carries a `<Scope>` badge saying
which — without it the page can look self-contradictory, since the baseline
occasionally beats the ensemble on an individual seed even though the 5-seed
mean favours the ensemble.

`MODELS` here is a hardcoded 5-name display list — see the "watch out for"
note below.

### `TeamCombobox.jsx`

A type-to-search `<input>` + listbox, not a native `<select>`, because the
`/teams` list is ~190 national sides and users type a name they already know
rather than browse. Behavior worth knowing before touching it:

- `value` is always a name the API actually serves; free text typed into the
  box is a *query* only, resolved back to a real team on Enter/Tab/blur/click
  via `teamSearch.searchTeams()`. If nothing resolves, the field snaps back to
  the last committed value — the point is to never let the form submit a name
  the backend will 400 on.
- `excluded` disables whichever team is already picked in the *other* combobox
  and keeps keyboard navigation (arrow keys) skipping over it.
- Query state (`query`) is `null` when showing the committed value and a
  string while actively searching — this distinction is what lets focus
  select-all the current value for easy replacement without it being
  mistaken for "the user searched for an empty string."

### `teamSearch.js`

Ranked (not just filtered) fuzzy matching, tiered `EXACT > PREFIX >
WORD_PREFIX > SUBSTRING > INITIALS > SUBSEQUENCE > TYPO` so the closest name
is always first and can be committed with Enter without scrolling. Handles
diacritics/punctuation (`normalizeWithMap`, which also tracks an index map
back to the original string for highlighting), colloquial/dataset-alternate
names (`ALIASES`, e.g. "Holland" → "Netherlands", "USA" → "United States"),
and typos (`withinEdits`, a length-scaled Levenshtein cutoff). If you add a
team-name normalization to `src/data/clean.py`'s `NAME_ALIASES`, consider
whether `teamSearch.js`'s `ALIASES` needs the same colloquial entry — the two
lists solve different problems (dataset-to-dataset joins vs. user typing) and
are not kept in sync automatically.

### `constants.js`

`MODEL_LABEL` is meant to be the single source of truth for the 5 canonical
classifier names. `humanizeFeature()` turns engineered feature names like
`movement_agility_b` or `weight_kg_diff` into display strings for the feature
importance chart, based on the `_a`/`_b`/`_diff` suffix convention shared with
`src/features/build_features.py`.

## Watch out for

- **The 5-model count is hardcoded in the frontend.** `Dashboard.jsx`'s
  `MODELS` array and the vote-count assumptions in `Predictor.jsx` assume
  exactly the 5 classifiers in `config.yaml`'s default `models.classifiers`.
  Changing that list on the backend without updating `Dashboard.jsx` will
  mislabel the "5-model ensemble" roster line (it won't crash — `MODELS` is
  just a display string, not driven by the API response).
- **Vite dev proxy assumes the backend is on `:8000`.** If you run the
  backend on a different port, update `vite.config.js`'s proxy target, not
  `api.js`.
- **`--strictPort` in `start.py`'s frontend launch** means the dev server
  fails loudly instead of silently moving to another port if `:5173` is
  taken — so the URL `start.py` prints and opens is always accurate.

## Local development

```bash
cd frontend
npm install
npm run dev      # Vite dev server, default http://localhost:5173
npm run lint      # oxlint
npm run build     # production build
```

See the root [README](../README.md#web-app) for the one-command
`python start.py` path that starts both the backend and frontend together.
