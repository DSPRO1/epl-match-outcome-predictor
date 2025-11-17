"""
EPL Match Outcome Predictor - Modal API Deployment

FastAPI inference endpoint deployed on Modal for real-time match predictions.

Usage:
    modal deploy modal_api.py
"""

import os
import pickle
from typing import Annotated, Optional

import modal
from fastapi import FastAPI, Header, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Define Modal app and image
app = modal.App("epl-predictor")

# Create container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "scikit-learn==1.5.2",
    "pandas==2.3.3",
    "numpy==1.26.4",
    "requests==2.32.5",
    "python-dateutil==2.9.0.post0",
    "fastapi[standard]",
    "psycopg2-binary==2.9.10",
)

# Create Modal volume for model storage
volume = modal.Volume.from_name("epl-models", create_if_missing=True)
MODEL_PATH = "/models/random_forest_model.pkl"
FEATURES_PATH = "/models/features.pkl"


class MatchInput(BaseModel):
    """Input schema for match prediction.

    Only team names are required. Stats will be looked up automatically from the database.
    All other fields are optional overrides for "what-if" scenarios.
    """

    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_elo: Optional[float] = Field(
        None, description="Home team ELO rating (optional override)"
    )
    away_elo: Optional[float] = Field(
        None, description="Away team ELO rating (optional override)"
    )
    home_gf_roll: Optional[float] = Field(
        None, description="Home team goals for (optional override)"
    )
    home_ga_roll: Optional[float] = Field(
        None, description="Home team goals against (optional override)"
    )
    home_pts_roll: Optional[float] = Field(
        None, description="Home team points (optional override)"
    )
    away_gf_roll: Optional[float] = Field(
        None, description="Away team goals for (optional override)"
    )
    away_ga_roll: Optional[float] = Field(
        None, description="Away team goals against (optional override)"
    )
    away_pts_roll: Optional[float] = Field(
        None, description="Away team points (optional override)"
    )
    rest_days_home: Optional[float] = Field(
        None, description="Days since home team last match (optional override)"
    )
    rest_days_away: Optional[float] = Field(
        None, description="Days since away team last match (optional override)"
    )


class PredictionOutput(BaseModel):
    """Output schema for match prediction."""

    home_team: str
    away_team: str
    prediction: str
    probabilities: dict
    confidence: float
    features_used: dict


class TeamStats(BaseModel):
    """Team statistics from database."""

    team_name: str
    elo_rating: float
    goals_for_avg: float
    goals_against_avg: float
    points_avg: float
    last_match_date: str
    matches_played: int


def get_db_connection():
    """Create database connection."""
    import psycopg2

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return psycopg2.connect(database_url)


def get_team_stats(team_name: str) -> dict:
    """Fetch team stats from database."""
    from datetime import datetime

    import psycopg2.extras

    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM team_stats WHERE team_name = %s", (team_name,))
        result = cur.fetchone()

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Team '{team_name}' not found in database. Please check team name spelling.",
            )

        # Calculate rest days
        last_match = result["last_match_date"]
        if last_match:
            rest_days = (datetime.now().date() - last_match).days
        else:
            rest_days = 7.0

        return {
            "elo_rating": float(result["elo_rating"]),
            "gf_roll": float(result["goals_for_avg"]),
            "ga_roll": float(result["goals_against_avg"]),
            "pts_roll": float(result["points_avg"]),
            "rest_days": float(rest_days),
        }
    finally:
        conn.close()


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=60,
)
def upload_model(model_bytes: bytes, features_bytes: bytes):
    """
    Upload a pre-trained model to Modal volume.

    Args:
        model_bytes: Pickled model bytes
        features_bytes: Pickled features list bytes
    """
    import pickle

    print("Uploading model to Modal volume...")

    # Save model and features
    with open(MODEL_PATH, "wb") as f:
        f.write(model_bytes)

    with open(FEATURES_PATH, "wb") as f:
        f.write(features_bytes)

    volume.commit()

    print("✓ Model uploaded successfully!")

    # Verify model can be loaded
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    return {
        "status": "success",
        "model_path": MODEL_PATH,
        "features": features,
        "model_type": str(type(model).__name__),
    }


# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key from request header."""
    correct_api_key = os.environ.get("EPL_API_KEY")

    if not correct_api_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if not api_key:
        raise HTTPException(
            status_code=401, detail="Missing API key. Include 'X-API-Key' header."
        )

    if api_key != correct_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


# Create FastAPI app
web_app = FastAPI(
    title="EPL Match Outcome Predictor API",
    description="Predict English Premier League match outcomes using machine learning",
    version="1.0.0",
)

# Add CORS middleware to allow web UI requests
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dspro1.zayden.ch",
        "https://epl-match-outcome-predictor.vercel.app",
        "http://localhost:4321",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "EPL Match Outcome Predictor",
        "status": "healthy",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict (POST, requires API key)",
            "teams": "/teams (GET, public)",
            "health": "/health (GET, public)",
            "docs": "/docs (GET, public)",
        },
        "note": "Team stats are now fetched automatically from database. Only team names are required for predictions.",
    }


@web_app.get("/health")
async def health():
    """Health check endpoint."""
    import os

    model_exists = os.path.exists(MODEL_PATH)
    return {"status": "healthy", "model_loaded": model_exists}


@web_app.post("/predict", response_model=PredictionOutput)
async def predict(match: MatchInput, api_key: str = Security(verify_api_key)):
    """
    Predict match outcome based on input features.

    Requires authentication via X-API-Key header.

    Returns probabilities for:
    - Home win or Draw (H_or_D)
    - Away win (A)
    """
    import os

    import numpy as np

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503, detail="Model not initialized. Please run training first."
        )

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    # Fetch team stats from database (or use provided overrides)
    home_stats = get_team_stats(match.home_team)
    away_stats = get_team_stats(match.away_team)

    # Use provided values if available, otherwise use database values
    home_elo = (
        match.home_elo if match.home_elo is not None else home_stats["elo_rating"]
    )
    away_elo = (
        match.away_elo if match.away_elo is not None else away_stats["elo_rating"]
    )
    home_gf_roll = (
        match.home_gf_roll if match.home_gf_roll is not None else home_stats["gf_roll"]
    )
    home_ga_roll = (
        match.home_ga_roll if match.home_ga_roll is not None else home_stats["ga_roll"]
    )
    home_pts_roll = (
        match.home_pts_roll
        if match.home_pts_roll is not None
        else home_stats["pts_roll"]
    )
    away_gf_roll = (
        match.away_gf_roll if match.away_gf_roll is not None else away_stats["gf_roll"]
    )
    away_ga_roll = (
        match.away_ga_roll if match.away_ga_roll is not None else away_stats["ga_roll"]
    )
    away_pts_roll = (
        match.away_pts_roll
        if match.away_pts_roll is not None
        else away_stats["pts_roll"]
    )
    rest_days_home = (
        match.rest_days_home
        if match.rest_days_home is not None
        else home_stats["rest_days"]
    )
    rest_days_away = (
        match.rest_days_away
        if match.rest_days_away is not None
        else away_stats["rest_days"]
    )

    # Prepare features
    goal_diff_pre = home_elo - away_elo
    rest_days_diff = rest_days_home - rest_days_away

    X = np.array(
        [
            [
                home_elo,
                away_elo,
                goal_diff_pre,
                home_gf_roll,
                home_ga_roll,
                home_pts_roll,
                away_gf_roll,
                away_ga_roll,
                away_pts_roll,
                rest_days_home,
                rest_days_away,
                rest_days_diff,
            ]
        ]
    )

    # Make prediction
    proba = model.predict_proba(X)[0]
    pred_label = int(np.argmax(proba))

    # Map prediction
    label_map = {0: "Home Win or Draw", 1: "Away Win"}
    prediction = label_map[pred_label]

    probabilities = {"home_or_draw": float(proba[0]), "away": float(proba[1])}

    confidence = float(max(proba))

    features_used = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "goal_diff_pre": goal_diff_pre,
        "home_gf_roll": home_gf_roll,
        "home_ga_roll": home_ga_roll,
        "home_pts_roll": home_pts_roll,
        "away_gf_roll": away_gf_roll,
        "away_ga_roll": away_ga_roll,
        "away_pts_roll": away_pts_roll,
        "rest_days_home": rest_days_home,
        "rest_days_away": rest_days_away,
        "rest_days_diff": rest_days_diff,
    }

    return PredictionOutput(
        home_team=match.home_team,
        away_team=match.away_team,
        prediction=prediction,
        probabilities=probabilities,
        confidence=confidence,
        features_used=features_used,
    )


@web_app.get("/teams")
async def list_teams():
    """
    List all available teams and their current statistics.

    No authentication required - public endpoint.
    """
    import psycopg2.extras

    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT
                team_name,
                elo_rating,
                goals_for_avg,
                goals_against_avg,
                points_avg,
                last_match_date,
                matches_played
            FROM team_stats
            ORDER BY elo_rating DESC
        """)
        teams = cur.fetchall()

        return {
            "count": len(teams),
            "teams": [
                {
                    "team_name": team["team_name"],
                    "elo_rating": float(team["elo_rating"]),
                    "goals_for_avg": float(team["goals_for_avg"]),
                    "goals_against_avg": float(team["goals_against_avg"]),
                    "points_avg": float(team["points_avg"]),
                    "last_match_date": str(team["last_match_date"]),
                    "matches_played": int(team["matches_played"]),
                }
                for team in teams
            ],
        }
    finally:
        conn.close()


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[
        modal.Secret.from_name("epl-api-secret"),
        modal.Secret.from_name("epl-database"),
    ],
)
@modal.asgi_app()
def fastapi_app():
    """Deploy FastAPI app on Modal with API key authentication."""
    return web_app
