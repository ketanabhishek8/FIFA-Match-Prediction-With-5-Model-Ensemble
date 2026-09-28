# Deployment

The web app deploys to **Vercel** as a single FastAPI function that serves both
the API and the built React frontend. The trained model it serves lives in
**Supabase Storage**, because neither the Kaggle training data nor the trained
bundle is in the repo, and training takes minutes — longer than a request
should wait.

```
Supabase Storage (public bucket)
└── <cache-key>.joblib          the trained model bundle, ~31 MB
        │  downloaded once per cold start, SHA-256 checked
        ▼
Vercel project
├── /            → frontend/dist   (built by vercel.json's buildCommand)
└── /api/*       → app.py → api/main.py (FastAPI)
```

## Pieces in the repo

| File | Role |
|---|---|
| `vercel.json` | FastAPI framework preset, the frontend build command, and `excludeFiles` keeping data, docs, tests and frontend sources out of the function bundle. |
| `app.py` | Root-level entrypoint Vercel looks for; re-exports `api.main:app`. |
| `.python-version` | Python 3.13, matching the environment that wrote the bundle. |
| `requirements.txt` | Runtime dependencies only — the only file Vercel installs from (~280 MB unpacked, against a 500 MB limit). Pinned exactly, because the bundle is a pickle. |
| `requirements-dev.txt` | Everything else, for local development. |
| `config/config.yaml` → `serving.model_bundle_sha256` | The SHA-256 the downloaded bundle must match. |

`api/model_cache.load_bundle()` downloads from `MODEL_BUNDLE_URL` when it is set
and falls back to the local train-or-load cache when it isn't, so local
development is unchanged. On Vercel with the variable missing, startup fails
with a message pointing here.

The API answers on both `/teams` and `/api/teams` (likewise `/predict`,
`/evaluation`): locally the Vite proxy strips `/api` before forwarding, while
on Vercel requests arrive with it.

## First deployment

1. **Supabase: create a public bucket.** In a Supabase project, go to
   *Storage → New bucket*, name it `models`, and turn on *Public bucket*.
   Public is fine here: the bundle holds fitted models and team-level
   aggregates, nothing secret, and integrity is enforced by the SHA-256 pin
   rather than by keeping the URL private.
2. **Upload the bundle.** Upload the file from `outputs/models/` whose SHA-256
   matches `serving.model_bundle_sha256` — currently
   `1cc48e7c3e9066d5.joblib`. (It exists only on a machine that has trained
   the model; see [DEVELOPMENT.md](DEVELOPMENT.md).) The Supabase free plan's
   50 MB upload limit covers it.
3. **Copy its public URL**, of the form
   `https://<project-ref>.supabase.co/storage/v1/object/public/models/1cc48e7c3e9066d5.joblib`.
4. **Vercel: set the environment variable.** *Project → Settings →
   Environment Variables*: add `MODEL_BUNDLE_URL` with that URL, for
   Production (and Preview, if you use preview deployments).
5. **Redeploy** — push to `main`, or *Deployments → ⋯ → Redeploy*. The
   project's *Root Directory* must be the repository root (the default).

## Updating the model

After a config change retrains the bundle locally:

```bash
shasum -a 256 outputs/models/<new-cache-key>.joblib
```

Upload the new file to the bucket, set `MODEL_BUNDLE_URL` to its URL, and put
its hash in `serving.model_bundle_sha256` in the same commit that changes the
config. Until the hash and the file agree, the deployed API refuses to start —
by design.

If you upgrade scikit-learn, xgboost, numpy or pandas in `requirements.txt`,
retrain and re-upload too: the pins exist because the bundle is only
guaranteed to load under the versions that wrote it.

## Behaviour to expect

- **Cold starts.** The first request after the function has been idle
  downloads, verifies and unpickles the bundle before answering — a few
  seconds. The frontend already treats a slow or 5xx backend as "still
  starting" and retries (see [FRONTEND.md](FRONTEND.md)).
- **Logs.** A failed download, a hash mismatch, or a missing
  `MODEL_BUNDLE_URL` shows up in the Vercel function logs as a
  `RuntimeError` naming the cause.
