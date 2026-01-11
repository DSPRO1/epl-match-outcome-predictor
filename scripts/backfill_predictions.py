#!/usr/bin/env python3
"""
Backfill predictions for matches that have already been played.

This script runs predictions on historical matches to populate the
predictions history table, allowing us to track model accuracy over time.

Usage:
    python scripts/backfill_predictions.py           # Interactive mode
    python scripts/backfill_predictions.py --yes    # Skip confirmation
    python scripts/backfill_predictions.py --limit 10  # Only process 10 matches
"""

import argparse
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR

# API Configuration
API_URL = os.environ.get("API_URL", "https://dspro1--epl-predictor-fastapi-app.modal.run")
API_KEY = os.environ.get("EPL_API_KEY", "")

MODELS = ["random_forest", "xgboost", "lightgbm"]


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


def run_prediction(home_team: str, away_team: str, model: str, match_date: str = None) -> dict:
    """Run a prediction via the API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    payload = {
        "home_team": home_team,
        "away_team": away_team,
        "model": model,
    }

    if match_date:
        payload["match_date"] = match_date

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error: {response.status_code} - {response.text[:100]}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Backfill predictions for played matches")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of matches to process")
    parser.add_argument("--model", "-m", type=str, default=None, help="Only use specific model (random_forest, xgboost, lightgbm)")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: EPL_API_KEY environment variable not set")
        print("Export it with: export EPL_API_KEY='your-api-key'")
        sys.exit(1)

    models_to_use = [args.model] if args.model else MODELS

    print("=" * 60)
    print("EPL Predictions Backfill Script")
    print("=" * 60)
    print(f"API: {API_URL}")
    print(f"Models: {', '.join(models_to_use)}")
    print()

    # Load matches
    matches = load_current_season_matches()

    if args.limit:
        matches = matches.head(args.limit)
        print(f"Limited to {args.limit} matches")

    if len(matches) == 0:
        print("No matches to process.")
        return

    # Show date range
    first_match = matches.iloc[0]
    last_match = matches.iloc[-1]
    print(f"Date range: {first_match['kickoff_datetime'].strftime('%Y-%m-%d')} to {last_match['kickoff_datetime'].strftime('%Y-%m-%d')}")
    print()

    # Ask for confirmation
    total_predictions = len(matches) * len(models_to_use)
    print(f"This will create {total_predictions} predictions ({len(matches)} matches x {len(models_to_use)} models)")

    if not args.yes:
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    print()
    print("Running predictions...")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, (_, match) in enumerate(matches.iterrows(), 1):
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["kickoff_datetime"].strftime("%Y-%m-%d")
        actual_score = f"{int(match['home_score'])}-{int(match['away_score'])}"

        print(f"[{idx}/{len(matches)}] {home_team} vs {away_team} ({match_date}) - Actual: {actual_score}")

        for model in models_to_use:
            result = run_prediction(home_team, away_team, model, match_date)

            if result:
                pred = result["prediction"]
                conf = result["confidence"] * 100
                print(f"  {model}: {pred} ({conf:.1f}% confidence)")
                successful += 1
            else:
                failed += 1

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        print()

    print("=" * 60)
    print(f"Backfill complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print()
    print(f"View predictions at: {API_URL}/predictions/history")


if __name__ == "__main__":
    main()
