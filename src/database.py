"""
Database operations module.

Handles PostgreSQL database connections and team statistics updates.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from .config import DATABASE_URL, ROLLING_WINDOW
from .elo import init_elo, compute_elo_update


def get_connection():
    """
    Create database connection.

    Returns:
        psycopg2 connection object

    Raises:
        ValueError: If DATABASE_URL not configured
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(DATABASE_URL)


def create_schema(conn):
    """
    Create team_stats table if it doesn't exist.

    Args:
        conn: Database connection
    """
    with open('schema.sql', 'r') as f:
        schema = f.read()

    cur = conn.cursor()
    cur.execute(schema)
    conn.commit()
    cur.close()


def calculate_team_stats_from_matches(matches_df: pd.DataFrame) -> List[Dict]:
    """
    Calculate current team statistics from historical match data.

    Args:
        matches_df: DataFrame with match history

    Returns:
        List of dictionaries with team statistics
    """
    print(f"Processing {len(matches_df)} matches...")

    # Ensure datetime column
    if 'kickoff_datetime' not in matches_df.columns:
        matches_df['kickoff_datetime'] = pd.to_datetime(matches_df['kickoff'])

    # Sort chronologically
    matches = matches_df.sort_values('kickoff_datetime').copy()

    # Initialize ELO ratings and match history
    teams = pd.concat([matches['home_team'], matches['away_team']]).unique()
    elo_ratings = init_elo(teams)
    match_history = defaultdict(list)

    # Process all matches to update ELO and history
    for _, row in matches.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        match_date = row['kickoff_datetime']

        # Get current ELO
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        # Update ELO
        new_home_elo, new_away_elo = compute_elo_update(
            home_score, away_score, home_elo, away_elo
        )
        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

        # Calculate points
        if home_score > away_score:
            home_pts, away_pts = 3, 0
        elif home_score < away_score:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        # Store match details
        match_history[home_team].append({
            'date': match_date,
            'gf': home_score,
            'ga': away_score,
            'pts': home_pts
        })
        match_history[away_team].append({
            'date': match_date,
            'gf': away_score,
            'ga': home_score,
            'pts': away_pts
        })

    # Calculate current stats for each team
    team_stats = []

    for team, history in match_history.items():
        if not history:
            continue

        # Get last N matches for rolling averages
        recent_matches = history[-ROLLING_WINDOW:]

        # Calculate averages
        gf_avg = sum(m['gf'] for m in recent_matches) / len(recent_matches)
        ga_avg = sum(m['ga'] for m in recent_matches) / len(recent_matches)
        pts_avg = sum(m['pts'] for m in recent_matches) / len(recent_matches)

        # Get last match date
        last_match_date = history[-1]['date']

        team_stats.append({
            'team_name': team,
            'elo_rating': round(elo_ratings[team], 2),
            'goals_for_avg': round(gf_avg, 2),
            'goals_against_avg': round(ga_avg, 2),
            'points_avg': round(pts_avg, 2),
            'last_match_date': last_match_date.date(),
            'matches_played': len(history),
            'updated_at': datetime.now(),
            'season': '2024-25'
        })

    print(f"Calculated stats for {len(team_stats)} teams")
    return team_stats


def update_team_stats(team_stats: List[Dict]):
    """
    Update PostgreSQL database with team statistics.

    Args:
        team_stats: List of team statistics dictionaries
    """
    print("Connecting to database...")
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Ensure schema exists
        print("Ensuring schema exists...")
        create_schema(conn)

        # Prepare data for bulk insert
        values = [
            (
                stat['team_name'],
                stat['elo_rating'],
                stat['goals_for_avg'],
                stat['goals_against_avg'],
                stat['points_avg'],
                stat['last_match_date'],
                stat['matches_played'],
                stat['updated_at'],
                stat['season']
            )
            for stat in team_stats
        ]

        # Upsert team stats
        print("Updating team stats...")
        insert_query = """
            INSERT INTO team_stats (
                team_name, elo_rating, goals_for_avg, goals_against_avg,
                points_avg, last_match_date, matches_played, updated_at, season
            ) VALUES %s
            ON CONFLICT (team_name)
            DO UPDATE SET
                elo_rating = EXCLUDED.elo_rating,
                goals_for_avg = EXCLUDED.goals_for_avg,
                goals_against_avg = EXCLUDED.goals_against_avg,
                points_avg = EXCLUDED.points_avg,
                last_match_date = EXCLUDED.last_match_date,
                matches_played = EXCLUDED.matches_played,
                updated_at = EXCLUDED.updated_at,
                season = EXCLUDED.season
        """

        execute_values(cur, insert_query, values)
        conn.commit()

        print(f"✓ Successfully updated {len(team_stats)} teams in database")

        # Display sample stats
        cur.execute("""
            SELECT team_name, elo_rating, goals_for_avg
            FROM team_stats
            ORDER BY elo_rating DESC
            LIMIT 5
        """)
        print("\nTop 5 teams by ELO:")
        for row in cur.fetchall():
            print(f"  {row[0]:<25} ELO: {row[1]:.0f}  Goals: {row[2]:.2f}")

    except Exception as e:
        conn.rollback()
        print(f"✗ Error updating database: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def get_team_stats(team_name: str) -> Dict:
    """
    Fetch team statistics from database.

    Args:
        team_name: Name of the team

    Returns:
        Dictionary with team statistics

    Raises:
        ValueError: If team not found
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM team_stats WHERE team_name = %s",
            (team_name,)
        )
        result = cur.fetchone()

        if not result:
            raise ValueError(f"Team '{team_name}' not found in database")

        # Calculate rest days
        last_match = result[5]  # last_match_date column
        if last_match:
            rest_days = (datetime.now().date() - last_match).days
        else:
            rest_days = 7.0

        return {
            'team_name': result[0],
            'elo_rating': float(result[1]),
            'goals_for_avg': float(result[2]),
            'goals_against_avg': float(result[3]),
            'points_avg': float(result[4]),
            'last_match_date': str(result[5]),
            'matches_played': int(result[6]),
            'rest_days': float(rest_days)
        }
    finally:
        conn.close()


def list_all_teams() -> List[Dict]:
    """
    List all teams in the database.

    Returns:
        List of team statistics dictionaries
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT team_name, elo_rating, goals_for_avg, goals_against_avg,
                   points_avg, last_match_date, matches_played
            FROM team_stats
            ORDER BY elo_rating DESC
        """)
        teams = []
        for row in cur.fetchall():
            teams.append({
                'team_name': row[0],
                'elo_rating': float(row[1]),
                'goals_for_avg': float(row[2]),
                'goals_against_avg': float(row[3]),
                'points_avg': float(row[4]),
                'last_match_date': str(row[5]),
                'matches_played': int(row[6])
            })
        return teams
    finally:
        conn.close()
