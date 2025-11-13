"""
Download EPL data from Premier League API and save to CSV

This script fetches all historical match data and saves it locally,
so you don't need to download it every time you train the model.

Usage:
    python download_data.py
"""

import requests
import pandas as pd
from datetime import date
import os


def download_data(output_dir='data'):
    """
    Download EPL match data from Premier League API.
    Uses the exact same method as colossal_sound.ipynb
    """
    print("="*80)
    print("Downloading EPL Match Data")
    print("="*80)

    season = 2025
    past_years = 11  # Download more years for better training
    start_matchweek = 1
    end_matchweek = 38

    all_matches = []
    all_standings = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\nFetching data for seasons {season-past_years+1}-{season}...")

    for year in range(past_years):
        current_season = season - year
        print(f"\nSeason {current_season}:")

        # Fetch standings
        standings_url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{current_season}/standings?live=false"

        try:
            response = requests.get(standings_url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"  ⚠ Standings Error {response.status_code}")
            else:
                standings = response.json()
                season_label = standings.get("season", {}).get("id", str(year))
                for entry in standings["tables"][0].get("entries", []):
                    team = entry.get("team", {}).get("name")
                    points = entry.get("overall", {}).get("points")
                    position = entry.get("overall", {}).get("position")
                    played = entry.get("overall", {}).get("played")
                    home_won = entry.get("home", {}).get("won")
                    home_lost = entry.get("home", {}).get("lost")
                    home_drawn = entry.get("home", {}).get("drawn")
                    home_goals_for = entry.get("home", {}).get("goalsFor")
                    home_goals_against = entry.get("home", {}).get("goalsAgainst")
                    away_won = entry.get("away", {}).get("won")
                    away_lost = entry.get("away", {}).get("lost")
                    away_drawn = entry.get("away", {}).get("drawn")
                    away_goals_for = entry.get("away", {}).get("goalsFor")
                    away_goals_against = entry.get("away", {}).get("goalsAgainst")
                    all_standings.append({
                        "season": season_label,
                        "position": position,
                        "team": team,
                        "points": points,
                        "played": played,
                        "home.won": home_won,
                        "home.lost": home_lost,
                        "home.drawn": home_drawn,
                        "home.goals_for": home_goals_for,
                        "home.goals_against": home_goals_against,
                        "away.won": away_won,
                        "away.lost": away_lost,
                        "away.drawn": away_drawn,
                        "away.goals_for": away_goals_for,
                        "away.goals_against": away_goals_against,
                    })
                print(f"  ✓ Standings loaded")

        except Exception as e:
            print(f"  ⚠ Standings Error: {str(e)}")

        # Determine matchweek range
        if current_season == season:
            season_2025_start_date = date(2025, 8, 15)
            actual_date = date.today()
            end_matchweek = min(38, round((actual_date - season_2025_start_date).days / 7))
        else:
            end_matchweek = 38

        # Fetch matches
        match_count = 0
        for matchweek in range(start_matchweek, end_matchweek + 1):
            matches_url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{current_season}/matchweeks/{matchweek}/matches"

            try:
                response = requests.get(matches_url, headers=headers, timeout=10)

                if response.status_code != 200:
                    continue

                data = response.json()
                matches_data = data.get('data', [])

                for match in matches_data:
                    try:
                        home_team = match.get('homeTeam', {})
                        away_team = match.get('awayTeam', {})

                        match_info = {
                            'match_id': match.get('matchId'),
                            'matchweek': matchweek,
                            'season': current_season,
                            'kickoff': match.get('kickoff'),
                            'kickoff_timezone': match.get('kickoffTimezone'),
                            'period': match.get('period'),
                            'competition': match.get('competition'),
                            'venue': match.get('ground'),
                            'attendance': match.get('attendance'),
                            'clock': match.get('clock'),
                            'result_type': match.get('resultType'),

                            # Home team info
                            'home_team': home_team.get('name'),
                            'home_team_id': home_team.get('id'),
                            'home_team_short': home_team.get('shortName'),
                            'home_score': home_team.get('score'),
                            'home_half_time_score': home_team.get('halfTimeScore'),
                            'home_red_cards': home_team.get('redCards'),

                            # Away team info
                            'away_team': away_team.get('name'),
                            'away_team_id': away_team.get('id'),
                            'away_team_short': away_team.get('shortName'),
                            'away_score': away_team.get('score'),
                            'away_half_time_score': away_team.get('halfTimeScore'),
                            'away_red_cards': away_team.get('redCards'),
                        }

                        # Determine match outcome
                        home_score = match_info['home_score']
                        away_score = match_info['away_score']

                        if home_score is not None and away_score is not None:
                            if home_score > away_score:
                                match_info['outcome'] = 'H'  # Home win
                                match_info['winner'] = match_info['home_team']
                            elif home_score < away_score:
                                match_info['outcome'] = 'A'  # Away win
                                match_info['winner'] = match_info['away_team']
                            else:
                                match_info['outcome'] = 'D'  # Draw
                                match_info['winner'] = 'Draw'
                            all_matches.append(match_info)
                            match_count += 1

                    except Exception as e:
                        continue

            except Exception as e:
                continue

        print(f"  ✓ Loaded {match_count} matches from matchweeks 1-{end_matchweek}")

    # Create DataFrames
    matches_df = pd.DataFrame(all_matches)
    standings_df = pd.DataFrame(all_standings)

    # Convert kickoff to datetime
    if 'kickoff' in matches_df.columns and len(matches_df) > 0:
        matches_df['kickoff_datetime'] = pd.to_datetime(matches_df['kickoff'], errors='coerce')

    print("\n" + "="*80)
    print(f"Total: {len(matches_df)} matches, {len(standings_df)} standings entries")
    print("="*80)

    if len(matches_df) > 0:
        print(f"\nSample data:")
        print(matches_df[['season', 'matchweek', 'home_team', 'away_team', 'home_score', 'away_score', 'outcome']].head(10))

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)

    matches_path = os.path.join(output_dir, 'matches.csv')
    standings_path = os.path.join(output_dir, 'standings.csv')

    matches_df.to_csv(matches_path, index=False)
    standings_df.to_csv(standings_path, index=False)

    print(f"\n✓ Data saved to:")
    print(f"  - {matches_path} ({len(matches_df)} matches)")
    print(f"  - {standings_path} ({len(standings_df)} standings)")

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Seasons: {matches_df['season'].min()} - {matches_df['season'].max()}")
    print(f"  Date range: {matches_df['kickoff_datetime'].min()} to {matches_df['kickoff_datetime'].max()}")
    print(f"  Total matches: {len(matches_df)}")
    print(f"  Home wins: {(matches_df['outcome'] == 'H').sum()} ({(matches_df['outcome'] == 'H').sum() / len(matches_df) * 100:.1f}%)")
    print(f"  Draws: {(matches_df['outcome'] == 'D').sum()} ({(matches_df['outcome'] == 'D').sum() / len(matches_df) * 100:.1f}%)")
    print(f"  Away wins: {(matches_df['outcome'] == 'A').sum()} ({(matches_df['outcome'] == 'A').sum() / len(matches_df) * 100:.1f}%)")

    return matches_df, standings_df


if __name__ == "__main__":
    download_data()
