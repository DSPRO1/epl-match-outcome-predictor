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
    "xgboost",
    "lightgbm",
)

# Create Modal volume for model storage
volume = modal.Volume.from_name("epl-models", create_if_missing=True)
MODEL_PATHS = {
    "random_forest": "/models/random_forest_model.pkl",
    "xgboost": "/models/xgboost_model.pkl",
    "lightgbm": "/models/lightgbm_model.pkl",
}
FEATURES_PATH = "/models/features.pkl"


class MatchInput(BaseModel):
    """Input schema for match prediction.

    Only team names are required. Stats will be looked up automatically from the database.
    All other fields are optional overrides for "what-if" scenarios.
    """

    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    model: Optional[str] = Field(
        "random_forest",
        description="Model to use for prediction: random_forest, xgboost, or lightgbm",
    )
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
    match_date: Optional[str] = Field(
        None, description="Match date (YYYY-MM-DD) for historical predictions"
    )


class PredictionOutput(BaseModel):
    """Output schema for match prediction."""

    home_team: str
    away_team: str
    prediction: str
    probabilities: dict
    confidence: float
    features_used: dict
    model_used: str


class TeamStats(BaseModel):
    """Team statistics from database."""

    team_name: str
    elo_rating: float
    goals_for_avg: float
    goals_against_avg: float
    points_avg: float
    last_match_date: str
    matches_played: int


class Fixture(BaseModel):
    """Upcoming fixture information."""

    match_id: int
    home_team: str
    away_team: str
    kickoff: str
    matchweek: int
    season: int
    venue: Optional[str] = None


def get_db_connection():
    """Create database connection."""
    import psycopg2

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return psycopg2.connect(database_url)


def get_h2h_stats(home_team: str, away_team: str) -> dict:
    """Calculate H2H stats from database."""
    import psycopg2.extras
    import psycopg2.errors

    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """
                SELECT home_team, away_team, home_score, away_score, kickoff_datetime
                FROM matches
                WHERE (
                    (home_team = %s AND away_team = %s) OR
                    (home_team = %s AND away_team = %s)
                )
                AND home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY kickoff_datetime DESC
                LIMIT 50
            """,
                (home_team, away_team, away_team, home_team),
            )

            past_matches = cur.fetchall()

            if not past_matches:
                return {"h2h_avg_points_home": 1.0, "h2h_avg_points_away": 1.0}

            home_points = []
            away_points = []

            for match in past_matches:
                if match["home_score"] > match["away_score"]:
                    points_home_team = 3
                    points_away_team = 0
                elif match["home_score"] < match["away_score"]:
                    points_home_team = 0
                    points_away_team = 3
                else:
                    points_home_team = 1
                    points_away_team = 1

                if match["home_team"] == home_team:
                    home_points.append(points_home_team)
                    away_points.append(points_away_team)
                else:
                    home_points.append(points_away_team)
                    away_points.append(points_home_team)

            return {
                "h2h_avg_points_home": sum(home_points) / len(home_points),
                "h2h_avg_points_away": sum(away_points) / len(away_points),
            }
        except psycopg2.errors.UndefinedTable:
            # matches table doesn't exist yet, return defaults
            return {"h2h_avg_points_home": 1.0, "h2h_avg_points_away": 1.0}
    finally:
        conn.close()


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


# Simple in-memory cache for fixtures (reset on deployment)
_fixtures_cache = {"data": None, "timestamp": None}
CACHE_DURATION_SECONDS = 300  # 5 minutes


def get_next_fixture() -> Optional[dict]:
    """
    Fetch the next upcoming EPL fixture from the Premier League API.
    Results are cached for 5 minutes to reduce API calls.
    """
    from datetime import datetime, timezone

    import requests

    # Check cache
    now = datetime.now(timezone.utc)
    if _fixtures_cache["data"] and _fixtures_cache["timestamp"]:
        cache_age = (now - _fixtures_cache["timestamp"]).total_seconds()
        if cache_age < CACHE_DURATION_SECONDS:
            return _fixtures_cache["data"]

    # Fetch from API
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Try current year and next year (EPL seasons span two years)
    current_year = datetime.now().year
    seasons_to_try = [current_year, current_year + 1, current_year - 1]

    all_upcoming = []

    for season in seasons_to_try:
        # Try all matchweeks for this season
        for matchweek in range(1, 39):
            try:
                url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{season}/matchweeks/{matchweek}/matches"
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    continue

                data = response.json()
                matches = data.get("data", [])

                # Find upcoming matches (no score yet)
                for match in matches:
                    home_team = match.get("homeTeam", {})
                    away_team = match.get("awayTeam", {})
                    kickoff = match.get("kickoff")

                    # Match hasn't been played if scores are None
                    if (
                        home_team.get("score") is None
                        and away_team.get("score") is None
                    ):
                        if kickoff:
                            try:
                                # Parse kickoff time and ensure it's timezone-aware (assume UTC)
                                if kickoff.endswith("Z"):
                                    kickoff_dt = datetime.fromisoformat(
                                        kickoff.replace("Z", "+00:00")
                                    )
                                else:
                                    # No timezone info, assume UTC
                                    kickoff_dt = datetime.fromisoformat(
                                        kickoff
                                    ).replace(tzinfo=timezone.utc)

                                # Only include future matches
                                if kickoff_dt > now:
                                    all_upcoming.append(
                                        {
                                            "match_id": match.get("matchId"),
                                            "home_team": home_team.get("name"),
                                            "away_team": away_team.get("name"),
                                            "kickoff": kickoff,
                                            "kickoff_dt": kickoff_dt,
                                            "matchweek": matchweek,
                                            "season": season,
                                            "venue": match.get("ground"),
                                        }
                                    )
                            except Exception:
                                continue

            except Exception:
                continue

        # If we found upcoming matches in this season, no need to check others
        if all_upcoming:
            break

    # Find the soonest match across all seasons
    if all_upcoming:
        next_match = min(all_upcoming, key=lambda x: x["kickoff_dt"])
        # Remove the datetime object before returning
        next_match.pop("kickoff_dt")

        # Cache the result
        _fixtures_cache["data"] = next_match
        _fixtures_cache["timestamp"] = now

        return next_match

    return None


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=60,
)
def upload_model(
    model_bytes: bytes, features_bytes: bytes, model_name: str = "random_forest"
):
    """
    Upload a pre-trained model to Modal volume.

    Args:
        model_bytes: Pickled model bytes
        features_bytes: Pickled features list bytes
        model_name: Name of the model being uploaded
    """
    import pickle

    print(f"Uploading {model_name} model to Modal volume...")

    # Save model and features
    model_path = MODEL_PATHS[model_name]
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    with open(FEATURES_PATH, "wb") as f:
        f.write(features_bytes)

    volume.commit()

    print("✓ Model uploaded successfully!")

    # Verify model can be loaded
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    return {
        "status": "success",
        "model_path": model_path,
        "features": features,
        "model_type": str(type(model).__name__),
        "model_name": model_name,
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
        "version": "2.2.0",
        "endpoints": {
            "predict": "/predict (POST, requires API key)",
            "teams": "/teams (GET, public)",
            "fixtures": "/fixtures/next (GET, public)",
            "predictions_history": "/predictions/history (GET, public)",
            "health": "/health (GET, public)",
            "docs": "/docs (GET, public)",
        },
        "note": "Team stats are now fetched automatically from database. Only team names are required for predictions.",
    }


@web_app.get("/health")
async def health():
    """Health check endpoint."""
    import os

    models_status = {model: os.path.exists(path) for model, path in MODEL_PATHS.items()}
    return {"status": "healthy", "models_loaded": models_status}


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
    import traceback

    import numpy as np

    try:
        # Validate model selection
        if match.model not in MODEL_PATHS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{match.model}'. Choose from: {list(MODEL_PATHS.keys())}",
            )

        # Check if model exists
        model_path = MODEL_PATHS[match.model]
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail=f"Model '{match.model}' not initialized. Please run training first.",
            )

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)

        # Fetch team stats from database (or use provided overrides)
        home_stats = get_team_stats(match.home_team)
        away_stats = get_team_stats(match.away_team)
        h2h_stats = get_h2h_stats(match.home_team, match.away_team)

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
        h2h_avg_points_home = h2h_stats["h2h_avg_points_home"]
        h2h_avg_points_away = h2h_stats["h2h_avg_points_away"]

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
                    h2h_avg_points_home,
                    h2h_avg_points_away,
                    rest_days_home,
                    rest_days_away,
                    rest_days_diff,
                ]
            ]
        )

        # Make prediction
        # LightGBM Booster objects use predict() directly, not predict_proba()
        if match.model == "lightgbm":
            raw_pred = model.predict(X)[0]
            # LightGBM returns probability of class 1 (Away Win)
            if isinstance(raw_pred, (int, float)):
                proba = np.array([1 - raw_pred, raw_pred])
            else:
                proba = raw_pred
        else:
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
            "h2h_avg_points_home": h2h_avg_points_home,
            "h2h_avg_points_away": h2h_avg_points_away,
            "rest_days_home": rest_days_home,
            "rest_days_away": rest_days_away,
            "rest_days_diff": rest_days_diff,
        }

        # Save prediction to database
        try:
            from psycopg2.extras import Json

            save_conn = get_db_connection()
            save_cur = save_conn.cursor()

            # Create table if not exists
            save_cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    home_team VARCHAR(255) NOT NULL,
                    away_team VARCHAR(255) NOT NULL,
                    prediction VARCHAR(50) NOT NULL,
                    home_or_draw_prob FLOAT NOT NULL,
                    away_prob FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    features JSONB,
                    model_used VARCHAR(50),
                    match_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            save_cur.execute(
                """
                INSERT INTO predictions (
                    home_team, away_team, prediction, home_or_draw_prob,
                    away_prob, confidence, features, model_used, match_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    match.home_team,
                    match.away_team,
                    prediction,
                    float(proba[0]),
                    float(proba[1]),
                    confidence,
                    Json(features_used),
                    match.model,
                    match.match_date,
                ),
            )
            save_conn.commit()
            save_cur.close()
            save_conn.close()
        except Exception as e:
            print(f"Warning: Failed to save prediction to database: {e}")

        return PredictionOutput(
            home_team=match.home_team,
            away_team=match.away_team,
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            features_used=features_used,
            model_used=match.model,
        )
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


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


@web_app.get("/fixtures/next", response_model=Fixture)
async def get_next_match():
    """
    Get the next upcoming EPL fixture.

    Returns the soonest upcoming match with kickoff time.
    Results are cached for 5 minutes.

    No authentication required - public endpoint.
    """
    fixture = get_next_fixture()

    if not fixture:
        raise HTTPException(
            status_code=404,
            detail="No upcoming fixtures found. Check back during the EPL season.",
        )

    return fixture


@web_app.get("/predictions/history")
async def get_prediction_history(limit: int = 50, offset: int = 0, model: str = None):
    """
    Get historical predictions with actual match results.

    Returns predictions made by the system along with whether
    they were correct based on actual match outcomes.

    No authentication required - public endpoint.

    Query params:
        limit: Maximum number of predictions to return (default 50)
        offset: Number of predictions to skip for pagination (default 0)
        model: Filter by model name (random_forest, xgboost, lightgbm)
    """
    import psycopg2.extras

    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Check if predictions table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'predictions'
            )
        """)
        table_exists = cur.fetchone()["exists"]

        if not table_exists:
            return {
                "count": 0,
                "total_with_results": 0,
                "correct_predictions": 0,
                "accuracy": None,
                "predictions": [],
                "message": "No predictions have been recorded yet"
            }

        # Check which columns exist
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'predictions'
        """)
        existing_columns = {row["column_name"] for row in cur.fetchall()}

        has_model_used = "model_used" in existing_columns
        has_match_date = "match_date" in existing_columns

        # Build query based on available columns
        select_cols = [
            "id", "home_team", "away_team", "prediction",
            "home_or_draw_prob", "away_prob", "confidence", "created_at"
        ]
        if has_model_used:
            select_cols.append("model_used")
        if has_match_date:
            select_cols.append("match_date")

        # Build WHERE clause with filters
        where_conditions = []
        query_params = []

        if has_match_date:
            where_conditions.append("match_date IS NOT NULL")

        if model and has_model_used:
            where_conditions.append("model_used = %s")
            query_params.append(model)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # Add limit and offset to params
        query_params.extend([min(limit, 100), offset])

        predictions_query = f"""
            SELECT {', '.join(select_cols)}
            FROM predictions
            {where_clause}
            ORDER BY match_date DESC, created_at DESC
            LIMIT %s OFFSET %s
        """

        # Get total count for pagination first
        count_conditions = [c for c in where_conditions]  # Copy conditions
        count_params = query_params[:-2]  # Exclude limit and offset
        count_where = "WHERE " + " AND ".join(count_conditions) if count_conditions else ""
        count_query = f"SELECT COUNT(*) FROM predictions {count_where}"
        cur.execute(count_query, count_params)
        total_count = cur.fetchone()["count"]

        # Now get the predictions
        cur.execute(predictions_query, query_params)
        prediction_rows = cur.fetchall()

        predictions = []
        for row in prediction_rows:
            pred = {
                "id": row["id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "prediction": row["prediction"],
                "home_or_draw_prob": float(row["home_or_draw_prob"]) if row["home_or_draw_prob"] else None,
                "away_prob": float(row["away_prob"]) if row["away_prob"] else None,
                "confidence": float(row["confidence"]) if row["confidence"] else None,
                "model_used": row.get("model_used") if has_model_used else None,
                "match_date": str(row["match_date"]) if has_match_date and row.get("match_date") else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "actual_home_score": None,
                "actual_away_score": None,
                "actual_result": None,
                "was_correct": None,
            }

            # Try to find matching result using match_date if available
            lookup_date = row.get("match_date") if has_match_date and row.get("match_date") else row["created_at"]
            if lookup_date:
                try:
                    match_query = """
                        SELECT home_score, away_score
                        FROM matches
                        WHERE home_team = %s AND away_team = %s
                        AND DATE(kickoff_datetime) = %s
                        AND home_score IS NOT NULL AND away_score IS NOT NULL
                        LIMIT 1
                    """
                    cur.execute(match_query, (row["home_team"], row["away_team"], str(lookup_date)[:10]))
                    match_result = cur.fetchone()

                    if match_result:
                        home_score = match_result["home_score"]
                        away_score = match_result["away_score"]

                        if home_score > away_score:
                            actual_result = "Home Win"
                        elif home_score < away_score:
                            actual_result = "Away Win"
                        else:
                            actual_result = "Draw"

                        if row["prediction"] == "Home Win or Draw":
                            was_correct = actual_result in ["Home Win", "Draw"]
                        elif row["prediction"] == "Away Win":
                            was_correct = actual_result == "Away Win"
                        else:
                            was_correct = None

                        pred["actual_home_score"] = home_score
                        pred["actual_away_score"] = away_score
                        pred["actual_result"] = actual_result
                        pred["was_correct"] = was_correct
                except Exception:
                    pass  # If matches table doesn't exist, continue without results

            predictions.append(pred)

        # Calculate OVERALL accuracy stats (across all predictions, not just current page)
        # Build stats where clause with table alias
        stats_conditions = []
        stats_params = []
        if has_match_date:
            stats_conditions.append("p.match_date IS NOT NULL")
        if model and has_model_used:
            stats_conditions.append("p.model_used = %s")
            stats_params.append(model)
        stats_where = "WHERE " + " AND ".join(stats_conditions) if stats_conditions else ""

        stats_query = f"""
            SELECT
                p.id, p.home_team, p.away_team, p.prediction, p.match_date,
                m.home_score, m.away_score
            FROM predictions p
            LEFT JOIN matches m ON p.home_team = m.home_team
                AND p.away_team = m.away_team
                AND DATE(m.kickoff_datetime) = p.match_date
            {stats_where}
        """
        cur.execute(stats_query, stats_params)
        all_predictions = cur.fetchall()

        total_with_results = 0
        correct_predictions = 0

        for row in all_predictions:
            if row["home_score"] is not None and row["away_score"] is not None:
                total_with_results += 1
                # Determine actual result
                if row["home_score"] > row["away_score"]:
                    actual = "Home Win"
                elif row["home_score"] < row["away_score"]:
                    actual = "Away Win"
                else:
                    actual = "Draw"

                # Check if prediction was correct
                if row["prediction"] == "Home Win or Draw":
                    if actual in ["Home Win", "Draw"]:
                        correct_predictions += 1
                elif row["prediction"] == "Away Win":
                    if actual == "Away Win":
                        correct_predictions += 1

        accuracy = (
            correct_predictions / total_with_results * 100
            if total_with_results > 0
            else None
        )

        return {
            "count": len(predictions),
            "total_count": total_count,
            "offset": offset,
            "has_more": offset + len(predictions) < total_count,
            "total_with_results": total_with_results,
            "correct_predictions": correct_predictions,
            "accuracy": round(accuracy, 2) if accuracy else None,
            "predictions": predictions,
        }
    except Exception as e:
        error_msg = str(e)
        # Handle case where predictions table doesn't exist yet
        if "relation \"predictions\" does not exist" in error_msg:
            return {
                "count": 0,
                "total_with_results": 0,
                "correct_predictions": 0,
                "accuracy": None,
                "predictions": [],
                "message": "No predictions have been recorded yet. Make some predictions and check back!"
            }
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch prediction history: {error_msg}"
        )
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
