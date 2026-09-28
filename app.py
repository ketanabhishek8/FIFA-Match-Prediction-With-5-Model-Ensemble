"""Vercel entrypoint. Vercel's FastAPI preset looks for an `app` in a root-level
`app.py`; the application itself lives in `api/main.py`. See docs/DEPLOYMENT.md.
"""
from api.main import app  # noqa: F401
