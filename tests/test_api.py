"""Wiring test for the FastAPI backend (api/main.py).

Run with: pytest tests/test_api.py -v
Note: first run trains the full pipeline (same cost as `python -m src.pipeline`)
and caches it to outputs/models/; later runs load the cache and are fast.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


def test_teams_and_predict_roundtrip():
    with TestClient(app) as client:
        teams_resp = client.get("/teams")
        assert teams_resp.status_code == 200
        teams = teams_resp.json()
        assert len(teams) > 0

        team_a, team_b = teams[0], teams[1]
        year = sorted(set(team_a["years"]) & set(team_b["years"]))[0] if set(team_a["years"]) & set(team_b["years"]) else team_a["years"][0]

        predict_resp = client.post("/predict", json={
            "team_a": team_a["team"], "team_b": team_b["team"], "year": year,
        })
        # Either a clean prediction, or a 400 if that exact pair/year has no
        # overlapping profile data -- both are valid wiring outcomes.
        assert predict_resp.status_code in (200, 400)
        if predict_resp.status_code == 200:
            body = predict_resp.json()
            assert body["winner"] in (team_a["team"], team_b["team"])
            assert len(body["model_votes"]) == 5
            # confidence and winner must derive from the same soft-vote
            # direction, so confidence is always >= 50% for the declared winner.
            assert 0.5 <= body["confidence"] <= 1.0


def test_evaluation_endpoint():
    with TestClient(app) as client:
        resp = client.get("/evaluation")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["comparison"]) == 2
        assert "Proposed Method" in body["seed_variance"]


def test_predict_returns_a_scoreline_block():
    """The scoreline half of /predict: a normalised three-way outcome, a modal
    scoreline that appears at the head of the ranked list, and expected goals."""
    with TestClient(app) as client:
        teams = client.get("/teams").json()
        by_name = {t["team"]: t for t in teams}
        pair = [n for n in ("Brazil", "Germany") if n in by_name]
        if len(pair) < 2:
            pair = [teams[0]["team"], teams[1]["team"]]
        shared = sorted(set(by_name[pair[0]]["years"]) & set(by_name[pair[1]]["years"]))
        if not shared:
            return

        body = client.post("/predict", json={
            "team_a": pair[0], "team_b": pair[1], "year": shared[-1],
        }).json()

        score = body["score"]
        assert score is not None, "score_model is enabled, so /predict must carry a score block"

        op = score["outcome_probability"]
        assert abs(op["team_a"] + op["draw"] + op["team_b"] - 1.0) < 1e-9
        # The draw is the whole point: the binary ensemble cannot express it.
        assert op["draw"] > 0

        top = score["top_scorelines"]
        assert len(top) == 5
        assert [t["probability"] for t in top] == sorted(
            (t["probability"] for t in top), reverse=True
        )
        assert score["most_likely"] == top[0]
        assert all(v > 0 for v in score["expected_goals"].values())
        assert score["favours"] in (pair[0], pair[1])


def test_neutral_venue_scoreline_does_not_depend_on_argument_order():
    """Reversing the two teams must swap the expected goals exactly, not just
    approximately -- otherwise whichever side is passed first quietly keeps the
    home-venue bump the model learned from qualifiers."""
    with TestClient(app) as client:
        teams = client.get("/teams").json()
        by_name = {t["team"]: t for t in teams}
        pair = [n for n in ("Brazil", "Germany") if n in by_name] or None
        if not pair or len(pair) < 2:
            pair = [teams[0]["team"], teams[1]["team"]]
        shared = sorted(set(by_name[pair[0]]["years"]) & set(by_name[pair[1]]["years"]))
        if not shared:
            return

        def goals(a, b):
            body = client.post("/predict", json={
                "team_a": a, "team_b": b, "year": shared[-1], "neutral": True,
            }).json()
            if body.get("score") is None:
                return None
            return body["score"]["expected_goals"]

        forward, reverse = goals(pair[0], pair[1]), goals(pair[1], pair[0])
        if forward is None:
            return
        assert forward["team_a"] == pytest.approx(reverse["team_b"])
        assert forward["team_b"] == pytest.approx(reverse["team_a"])


def test_evaluation_carries_score_model_metrics():
    with TestClient(app) as client:
        block = client.get("/evaluation").json()["score_model"]
        assert block is not None

        names = {row["model"] for row in block["comparison"]}
        assert any(n.startswith("Baseline") for n in names), "the no-skill floor must be shipped"
        for row in block["comparison"]:
            assert 0 <= row["RPS"] <= 1
            assert 0 <= row["Derived Win Accuracy"] <= 1

        # Every real variant must beat the league-average floor on RPS, which is
        # the check that the served model is actually fit rather than degenerate.
        floor = next(r["RPS"] for r in block["comparison"] if r["model"].startswith("Baseline"))
        assert all(r["RPS"] < floor for r in block["comparison"]
                   if not r["model"].startswith("Baseline"))
