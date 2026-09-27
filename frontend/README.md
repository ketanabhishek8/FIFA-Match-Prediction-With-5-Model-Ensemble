# Frontend

The FIFA match predictor's web app: a React + Vite single-page app with a live
team-vs-team predictor and an evaluation-results dashboard, talking to the
FastAPI backend in `../api/` through the Vite dev proxy.

**Documentation lives in [../docs/FRONTEND.md](../docs/FRONTEND.md)** —
component map, data flow, the `withRetry` cold-start behaviour, and what to
watch out for when changing things.

```bash
npm install
npm run dev      # Vite dev server, default http://localhost:5173
npm run lint     # oxlint
npm run build    # production build
```

The dev server proxies `/api/*` to `http://localhost:8000`, so the backend has
to be running too — `python start.py` from the repo root starts both. Needs
Node ^20.19 or >= 22.12 (Vite 8's floor).
