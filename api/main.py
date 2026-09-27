"""FastAPI backend for the FIFA match predictor frontend.

Run with: uvicorn api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, roc_curve

from api.model_cache import load_or_train
from src.config import load_config
from src.features.build_features import build_single_match_features
from src.models.score_model import outcome_proba_from_grid, scoreline_grid

SEEDS = [0, 1, 17, 42, 123]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    if cfg["project"]["random_state"] not in SEEDS:
        raise ValueError(
            f"config.yaml project.random_state ({cfg['project']['random_state']}) must be "
            f"one of the hardcoded evaluation SEEDS {SEEDS} (api/main.py) so a default "
            "report exists to serve -- add it to SEEDS or pick one of the existing values."
        )
    bundle = load_or_train(cfg, SEEDS)
    state["profiles"] = bundle["profiles"]
    state["reports_by_seed"] = bundle["reports_by_seed"]
    state["default_seed"] = bundle["default_seed"]
    state["default_report"] = bundle["reports_by_seed"][bundle["default_seed"]]
    state["history_state"] = bundle.get("history_state")

    score_reports = bundle.get("score_reports_by_seed") or {}
    state["score_reports_by_seed"] = score_reports
    # None when score_model.enabled is false, or when an older cached bundle
    # predates the scoreline model -- every consumer below treats it as optional
    # rather than assuming it exists.
    state["score_report"] = score_reports.get(bundle["default_seed"])
    yield


app = FastAPI(title="FIFA Match Predictor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    team_a: str
    team_b: str
    # Which FIFA edition's squad ratings to use. Omit for the most recent
    # edition both teams appear in -- that's what a *future* fixture needs,
    # since there is no player dataset for a tournament that hasn't happened.
    year: int | None = None
    # World Cup finals are played at neutral venues, but the model is trained
    # on home_team/away_team rows where, across all in-scope competitions with
    # draws dropped, 62.8% of non-neutral matches are home wins (vs 49.6% on
    # neutral ground). It therefore favours whichever team is
    # passed as team_a. With neutral=True the match is scored in both
    # orientations and averaged, so the answer no longer depends on argument
    # order. Set False only for a fixture with a genuine home side.
    neutral: bool = True


def _latest_shared_year(team_a: str, team_b: str) -> int:
    """Most recent edition for which BOTH teams have a squad profile."""
    profiles = state["profiles"]
    years_a = set(profiles.loc[profiles["team"] == team_a, "year"])
    years_b = set(profiles.loc[profiles["team"] == team_b, "year"])
    shared = years_a & years_b
    if not shared:
        raise HTTPException(
            status_code=400,
            detail=f"No FIFA edition has squad data for both {team_a!r} and {team_b!r}.",
        )
    return int(max(shared))


@app.get("/teams")
def teams() -> list[dict]:
    profiles = state["profiles"]
    by_team: dict[str, list[int]] = {}
    for team, year in zip(profiles["team"], profiles["year"]):
        by_team.setdefault(team, []).append(int(year))
    return [{"team": t, "years": sorted(years)} for t, years in sorted(by_team.items())]


def _proba_home_win(team_a: str, team_b: str, year: int, neutral: bool = True) -> dict[str, float]:
    """Per-model P(team_a wins) for one orientation of a matchup."""
    attrs = state["default_report"].attrs
    try:
        row = build_single_match_features(
            team_a, team_b, year, state["profiles"], state.get("history_state"),
            neutral=neutral,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = row[attrs["feature_cols"]].fillna(attrs["train_mean"])
    X_scaled = attrs["scaler"].transform(X)
    X_final = attrs["pca"].transform(X_scaled) if attrs["pca"] is not None else X_scaled

    ensemble = attrs["ensemble"]
    return {name: float(est.predict_proba(X_final)[0][1])
            for name, est in ensemble.named_estimators_.items()}


def _goal_rates(team_a: str, team_b: str, year: int, neutral: bool) -> tuple[float, float]:
    """Expected goals for (team_a, team_b) in one orientation, from the
    scoreline model. Uses the score model's OWN scaler/PCA/train_mean: it was
    fit on a draws-inclusive frame, so the classifier's transforms describe a
    different training population."""
    attrs = state["score_report"].attrs
    try:
        row = build_single_match_features(
            team_a, team_b, year, state["profiles"], state.get("history_state"),
            neutral=neutral,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = row[attrs["feature_cols"]].fillna(attrs["train_mean"])
    X_scaled = attrs["scaler"].transform(X)
    X_final = attrs["pca"].transform(X_scaled) if attrs["pca"] is not None else X_scaled

    lam, mu = attrs["model"].expected_goals(X_final)[0]
    return float(lam), float(mu)


def _score_prediction(team_a: str, team_b: str, year: int, neutral: bool,
                      ensemble_winner: str) -> dict:
    """Scoreline block for /predict: expected goals, the likeliest scorelines,
    and three-way outcome probabilities."""
    model = state["score_report"].attrs["model"]

    rate_a, rate_b = _goal_rates(team_a, team_b, year, neutral)
    if neutral:
        # Same argument-order fix the ensemble uses: score the reversed fixture
        # and average, so neither side keeps the home-venue bump the model
        # learned from qualifiers. Reversed, team_a is the AWAY side, so its
        # rate is the second returned value.
        rev_b, rev_a = _goal_rates(team_b, team_a, year, neutral)
        rate_a, rate_b = (rate_a + rev_a) / 2, (rate_b + rev_b) / 2

    grid = scoreline_grid(np.array([rate_a]), np.array([rate_b]),
                          max_goals=model.max_goals, rho=model.rho_)
    probs = outcome_proba_from_grid(grid)[0]

    width = grid.shape[1]
    flat = grid[0].reshape(-1)
    top_idx = np.argsort(flat)[::-1][:5]
    top = [{"team_a": int(i // width), "team_b": int(i % width), "probability": float(flat[i])}
           for i in top_idx]

    # Compare like with like: the ensemble has no draw class, so agreement is
    # judged on which side the score model favours to WIN, ignoring draw mass.
    # A disagreement is legitimate (the two models are fit to different targets)
    # and the frontend says so rather than quietly showing both.
    score_favours = team_a if probs[0] >= probs[2] else team_b

    return {
        "expected_goals": {"team_a": rate_a, "team_b": rate_b},
        "most_likely": top[0],
        "top_scorelines": top,
        "outcome_probability": {
            "team_a": float(probs[0]), "draw": float(probs[1]), "team_b": float(probs[2]),
        },
        "favours": score_favours,
        "agrees_with_ensemble": score_favours == ensemble_winner,
        # Dixon-Coles rho, so a client can tell whether the correction is on.
        "dixon_coles_rho": float(model.rho_),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if req.team_a == req.team_b:
        raise HTTPException(status_code=400, detail="Pick two different teams.")

    year = req.year if req.year is not None else _latest_shared_year(req.team_a, req.team_b)

    per_model_proba_a = _proba_home_win(req.team_a, req.team_b, year, req.neutral)
    if req.neutral:
        # Score the reversed fixture too and average. P(a wins) from the
        # reversed run is 1 - P(b wins as home), so the home-side bias the model
        # learned cancels instead of landing on whichever team was passed first.
        reversed_proba_b = _proba_home_win(req.team_b, req.team_a, year, req.neutral)
        per_model_proba_a = {
            name: (p + (1.0 - reversed_proba_b[name])) / 2.0
            for name, p in per_model_proba_a.items()
        }

    votes = {name: (req.team_a if p >= 0.5 else req.team_b) for name, p in per_model_proba_a.items()}
    # Each model's confidence in its OWN pick (always >= 0.5).
    model_confidence = {name: (p if p >= 0.5 else 1 - p) for name, p in per_model_proba_a.items()}

    # Winner and confidence both derive from the same soft-vote average, so
    # they can never disagree in direction (a hard-vote winner combined with a
    # separately-computed soft-vote confidence can point different ways when
    # the vote is a narrow, low-conviction majority). The offline evaluation
    # figures (README/report) still use the paper's hard majority vote via
    # ensemble.predict() -- this only affects the live single-match endpoint.
    mean_proba_a = sum(per_model_proba_a.values()) / len(per_model_proba_a)
    winner_is_a = mean_proba_a >= 0.5
    confidence = mean_proba_a if winner_is_a else 1 - mean_proba_a

    winner = req.team_a if winner_is_a else req.team_b

    return {
        "team_a": req.team_a,
        "team_b": req.team_b,
        # The edition actually used, which may differ from what was requested
        # (or have been chosen here when the caller omitted it).
        "year": year,
        "neutral": req.neutral,
        "winner": winner,
        "confidence": confidence,
        "model_votes": votes,
        "model_confidence": model_confidence,
        # Null when score_model.enabled is false or the cached bundle predates
        # the scoreline model, so the frontend degrades to the winner-only view.
        "score": (_score_prediction(req.team_a, req.team_b, year, req.neutral, winner)
                  if state.get("score_report") is not None else None),
    }


def _score_evaluation() -> dict | None:
    """Scoreline metrics for the dashboard. None when the model is disabled."""
    report = state.get("score_report")
    if report is None:
        return None

    by_seed = state["score_reports_by_seed"]
    rows = report.reset_index(names="model").to_dict(orient="records")
    fitted = report.attrs["model"]

    return {
        # One row per variant: the fitted model, the same model without the
        # Dixon-Coles correction, and the league-average floor. Shipping the
        # uncorrected row means the correction's (small) effect stays visible
        # rather than being an unverifiable claim in a config comment.
        "comparison": rows,
        "seed_variance": {
            metric: [
                {"seed": int(s), **{name: float(by_seed[s].loc[name, metric])
                                    for name in by_seed[s].index}}
                for s in SEEDS if s in by_seed
            ]
            for metric in ("RPS", "Derived Win Accuracy")
        },
        "dixon_coles_rho": float(report.attrs["rho"]),
        "dixon_coles_enabled": bool(fitted.dixon_coles),
        "n_train": int(report.attrs["n_train"]),
        "n_test": int(report.attrs["n_test"]),
    }


@app.get("/evaluation")
def evaluation() -> dict:
    report = state["default_report"]
    attrs = report.attrs

    seed_variance = {
        label: [
            {"seed": s, "accuracy": float(state["reports_by_seed"][s].loc[label, "Overall Accuracy"])}
            for s in SEEDS
        ]
        for label in ("Proposed Method", "Baseline Model")
    }

    pca_variance = attrs["pca_explained_variance"]

    fpr, tpr, _ = roc_curve(attrs["y_test"], attrs["ml_proba"])
    roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]

    return {
        "comparison": report.reset_index().to_dict(orient="records"),
        "seed_variance": seed_variance,
        "confusion_matrix": {
            "proposed": confusion_matrix(attrs["y_test"], attrs["ml_pred"]).tolist(),
            "baseline": confusion_matrix(attrs["y_test"], attrs["baseline_pred"]).tolist(),
        },
        "pca_cumulative_variance": np.cumsum(pca_variance).tolist() if pca_variance is not None else None,
        "per_model_accuracy": attrs["per_model_accuracy"].to_dict(),
        "feature_importance": attrs["feature_importance"].to_dict() if attrs["feature_importance"] is not None else None,
        "roc": {"auc": float(attrs["ensemble_auc"]), "points": roc_points},
        # Every block above except `seed_variance` comes from ONE seed's report
        # (the default seed). The frontend labels those panels with this value so
        # they can't be mistaken for the across-seed averages in `seed_variance`
        # -- on some seeds the baseline beats the ensemble even though the mean
        # goes the other way.
        "default_seed": int(state["default_seed"]),
        "seeds": [int(s) for s in SEEDS],
        # Null when score_model.enabled is false; the dashboard hides its panel.
        "score_model": _score_evaluation(),
        "dataset": {
            "n_team_profiles": int(len(state["profiles"])),
            "n_train_matches": int(attrs["n_train"]),
            "n_test_matches": int(attrs["n_test"]),
            "n_raw_features": int(len(attrs["feature_cols"])),
            "n_pca_components": int(len(pca_variance)) if pca_variance is not None else None,
        },
    }
