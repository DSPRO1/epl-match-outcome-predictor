#!/usr/bin/env python3
"""
Local backfill predictions for matches that have already been played.

Runs predictions locally using the trained models, much faster than API calls.

Usage:
    python scripts/backfill_local.py           # Interactive mode
    python scripts/backfill_local.py --yes     # Skip confirmation
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RANDOM_FOREST_MODEL, XGBOOST_MODEL, LIGHTGBM_MODEL, FEATURES_FILE
from src.database import get_connection

MODELS = {
    "random_forest": RANDOM_FOREST_MODEL,
    "xgboost": XGBOOST_MODEL,
    "lightgbm": LIGHTGBM_MODEL,
}


def load_models():
    """Load all available models."""
    loaded = {}
    for name, path in MODELS.items():
        if path.exists():
            print(f"Loading {name}...")
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
        else:
            print(f"Warning: {name} model not found at {path}")
    return loaded


def load_features():
    """Load feature list."""
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    with open(FEATURES_FILE, "rb") as f:
        return pickle.load(f)


def load_team_stats():
    """Load team stats from database."""
    conn = get_connection()
    try:
        df = pd.read_sql("SELECT * FROM team_stats", conn)
        return df.set_index("team_name").to_dict("index")
    finally:
        conn.close()


def get_features_for_match(home_team, away_team, team_stats):
    """Build feature vector for a match."""
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})

    home_elo = float(home.get("elo_rating", 1500))
    away_elo = float(away.get("elo_rating", 1500))

    return {
        "elo_home_pre": home_elo,
        "elo_away_pre": away_elo,
        "goal_diff_pre": home_elo - away_elo,
        "home_gf_roll": float(home.get("goals_for_avg", 1.5)),
        "home_ga_roll": float(home.get("goals_against_avg", 1.0)),
        "home_pts_roll": float(home.get("points_avg", 1.5)),
        "away_gf_roll": float(away.get("goals_for_avg", 1.5)),
        "away_ga_roll": float(away.get("goals_against_avg", 1.0)),
        "away_pts_roll": float(away.get("points_avg", 1.5)),
        "h2h_avg_points_home": 1.0,  # Default, no H2H data
        "h2h_avg_points_away": 1.0,
        "rest_days_home": 7.0,
        "rest_days_away": 7.0,
        "rest_days_diff": 0.0,
    }


def predict_match(model, features, feature_names, model_name):
    """Make prediction for a single match."""
    X = np.array([[features[f] for f in feature_names]])

    if model_name == "lightgbm":
        # LightGBM Booster uses predict() directly
        proba = model.predict(X)[0]
        if isinstance(proba, (int, float)):
            # Binary output
            proba = [1 - proba, proba]
    else:
        proba = model.predict_proba(X)[0]

    prediction = "Home Win or Draw" if proba[0] > proba[1] else "Away Win"
    confidence = max(proba[0], proba[1])

    return prediction, proba, confidence


def save_prediction(conn, prediction_data):
    """Save prediction to database."""
    from psycopg2.extras import Json

    cur = conn.cursor()

    # Ensure table exists
    cur.execute("""
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

    cur.execute(
        """
        INSERT INTO predictions (
            home_team, away_team, prediction, home_or_draw_prob,
            away_prob, confidence, features, model_used, match_date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            prediction_data["home_team"],
            prediction_data["away_team"],
            prediction_data["prediction"],
            prediction_data["home_or_draw_prob"],
            prediction_data["away_prob"],
            prediction_data["confidence"],
            Json(prediction_data.get("features")),
            prediction_data["model_used"],
            prediction_data["match_date"],
        ),
    )
    conn.commit()
    cur.close()


def load_current_season_matches():
    """Load matches from the current season that have been played."""
    matches_path = DATA_DIR / "matches.csv"

    if not matches_path.exists():
        print(f"Error: {matches_path} not found. Run pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(matches_path)

    # Filter for current season (2025) and matches with scores
    current_season = df[
        (df["season"] == 2025) &
        (df["home_score"].notna()) &
        (df["away_score"].notna())
    ].copy()

    # Sort by date
    current_season["kickoff_datetime"] = pd.to_datetime(current_season["kickoff_datetime"])
    current_season = current_season.sort_values("kickoff_datetime")

    print(f"Found {len(current_season)} played matches in the 2024-25 season")
    return current_season


def main():
    parser = argparse.ArgumentParser(description="Local backfill predictions")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit matches")
    parser.add_argument("--model", "-m", type=str, default=None, help="Specific model only")
    args = parser.parse_args()

    print("=" * 60)
    print("EPL Local Predictions Backfill")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    models = load_models()
    if not models:
        print("Error: No models available")
        sys.exit(1)

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not available")
            sys.exit(1)
        models = {args.model: models[args.model]}

    print(f"Models: {', '.join(models.keys())}")

    # Load features
    print("\nLoading features...")
    feature_names = load_features()
    print(f"Features: {len(feature_names)}")

    # Load team stats
    print("\nLoading team stats...")
    team_stats = load_team_stats()
    print(f"Teams: {len(team_stats)}")

    # Load matches
    print("\nLoading matches...")
    matches = load_current_season_matches()

    if args.limit:
        matches = matches.head(args.limit)
        print(f"Limited to {args.limit} matches")

    if len(matches) == 0:
        print("No matches to process.")
        return

    # Show summary
    total_predictions = len(matches) * len(models)
    print(f"\nThis will create {total_predictions} predictions ({len(matches)} matches x {len(models)} models)")

    if not args.yes:
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    # Connect to database
    print("\nConnecting to database...")
    conn = get_connection()

    print("\nRunning predictions...")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, (_, match) in enumerate(matches.iterrows(), 1):
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["kickoff_datetime"].strftime("%Y-%m-%d")
        actual_score = f"{int(match['home_score'])}-{int(match['away_score'])}"

        # Skip if team not in stats
        if home_team not in team_stats or away_team not in team_stats:
            print(f"[{idx}/{len(matches)}] {home_team} vs {away_team} - SKIPPED (team not found)")
            failed += 1
            continue

        print(f"[{idx}/{len(matches)}] {home_team} vs {away_team} ({match_date}) - Actual: {actual_score}")

        # Get features
        features = get_features_for_match(home_team, away_team, team_stats)

        for model_name, model in models.items():
            try:
                prediction, proba, confidence = predict_match(model, features, feature_names, model_name)

                # Save to database
                save_prediction(conn, {
                    "home_team": home_team,
                    "away_team": away_team,
                    "prediction": prediction,
                    "home_or_draw_prob": float(proba[0]),
                    "away_prob": float(proba[1]),
                    "confidence": float(confidence),
                    "features": features,
                    "model_used": model_name,
                    "match_date": match_date,
                })

                print(f"  {model_name}: {prediction} ({confidence*100:.1f}%)")
                successful += 1

            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")
                failed += 1

    conn.close()

    print("\n" + "=" * 60)
    print("Backfill complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
