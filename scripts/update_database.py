"""
Update team statistics in PostgreSQL database.

This script:
1. Loads historical match data from CSV
2. Calculates current ELO ratings and rolling statistics
3. Updates the PostgreSQL database with current team stats

Usage:
    export DATABASE_URL=<your_database_url>
    python scripts/update_database.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import MATCHES_CSV
from src.database import calculate_team_stats_from_matches, update_team_stats


def main():
    """Main execution function."""
    print("=" * 60)
    print("EPL TEAM STATS UPDATE")
    print("=" * 60)

    # Load match data
    print("\nLoading match data...")
    if not MATCHES_CSV.exists():
        print(f"✗ Data file not found: {MATCHES_CSV}")
        print("  Run 'python scripts/download_data.py' first")
        sys.exit(1)

    matches = pd.read_csv(MATCHES_CSV)
    matches['kickoff_datetime'] = pd.to_datetime(matches['kickoff_datetime'])
    print(f"Loaded {len(matches)} matches")

    # Calculate team stats
    team_stats = calculate_team_stats_from_matches(matches)

    # Update database
    update_team_stats(team_stats)

    print("\n" + "=" * 60)
    print("Update complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
