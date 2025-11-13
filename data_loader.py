import requests
import pandas as pd
from datetime import date



def load_data(*args, **kwargs):
    """
    Load EPL match outcomes from Premier League API.

    """

    season = 2025
    past_years = 11
    start_matchweek = 1
    end_matchweek = 38

    all_matches = []
    all_standings = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }


    for year in range(past_years):
        standings_url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{season-year}/standings?live=false"

        try:
            response = requests.get(standings_url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"Standings {year}: Error {response.status_code}")
                continue

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

        except Exception as e:
            print(f"Standings {year}: Error - {str(e)}")
            continue

        if season - year == season:
            season_2025_start_date = date(2025, 8, 15)
            actual_date = date.today()
            end_matchweek = round((actual_date - season_2025_start_date).days / 7) - 1
        else:
            end_matchweek = 38

        for matchweek in range(start_matchweek, end_matchweek + 1):
            matches_url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{season-year}/matchweeks/{matchweek}/matches"

            try:
                response = requests.get(matches_url, headers=headers, timeout=10)

                if response.status_code != 200:
                    print(f"Matchweek {matchweek}: Error {response.status_code}")
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
                            'season': season-year,
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
                                match_info['outcome'] = 'H_or_D'  # Home win
                                match_info['winner'] = match_info['home_team']
                            elif home_score < away_score:
                                match_info['outcome'] = 'A'  # Away win
                                match_info['winner'] = match_info['away_team']
                            else:
                                match_info['outcome'] = 'H_or_D'  # Draw
                                match_info['winner'] = 'Draw'
                            all_matches.append(match_info)

                    except Exception as e:
                        print(f"Error processing match in matchweek {matchweek}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Matchweek {matchweek}: Error - {str(e)}")
                continue

    matches_df = pd.DataFrame(all_matches)

    # Convert kickoff to datetime
    if 'kickoff' in matches_df.columns and len(matches_df) > 0:
        matches_df['kickoff_datetime'] = pd.to_datetime(matches_df['kickoff'], errors='coerce')

    print(f"\n✓ Successfully loaded {len(matches_df)} matches from matchweeks {start_matchweek}-{end_matchweek}")
    if len(matches_df) > 0:
        print(f"\nSample data:")
        print(matches_df[['matchweek', 'home_team', 'away_team', 'home_score', 'away_score', 'outcome']].head(5))

    standings_df = pd.DataFrame(all_standings)

    matches_df = adjust_values(matches_df)

    return matches_df, standings_df

def adjust_values(m):
    m['away_team_id'] = m['away_team_id'].astype(int)
    m['home_team_id'] = m['home_team_id'].astype(int)
    print("\n--- Basic Info ---")
    print(m.info())

    print("\n--- Sample Data ---")
    print(m.head())

    print("\n--- Missing Values ---")
    print(m.isna().sum())

    print("\n--- Duplicate Rows ---")
    print(m.duplicated().sum())

    print("\n--- Data Type Summary ---")
    print(m.dtypes)

    print("\n--- Numeric Range Checks ---")
    print("Attendance < 0:", (m['attendance'] < 0).sum())
    print("Goals negative:", ((m['away_score'] < 0) | (m['home_score'] < 0)).sum())
    print("Home Halftime Goals > Home Goals :", (m['home_half_time_score'] > m['home_score']).sum())
    print("Away Halftime Goals > Away Goals :", (m['away_half_time_score'] > m['away_score']).sum())
    print("Home Redcards < 0 :", (m['home_red_cards'] < 0).sum())
    print("Away Redcards < 0 :", (m['away_red_cards'] < 0).sum())

    print("\n--- Logical Checks ---")
    def winner_correct(row):
        if row['outcome'] == 'H' and row['winner'] != row['home_team']:
            return False
        elif row['outcome'] == 'A' and row['winner'] != row['away_team']:
            return False
        elif row['outcome'] == 'D' and row['winner'] != 'Draw':
            return False
        return True

    logic_issues = m.apply(lambda r: not winner_correct(r), axis=1).sum()
    print(f"Rows with mismatched Winner/Outcome: {logic_issues}")
    print("Missing matches:", 10 - len(m) % 10)

    print("\n--- Attendance Outliers ---")
    q1 = m['attendance'].quantile(0.25)
    q3 = m['attendance'].quantile(0.75)
    iqr = q3 - q1
    outliers = m[(m['attendance'] < (q1 - 1.5 * iqr)) | (m['attendance'] > (q3 + 1.5 * iqr))]
    print(f"Outlier count: {len(outliers)}")

    summary = {
        "rows": len(m),
        "columns": len(m.columns),
        "missing_values": m.isna().sum().sum(),
        "duplicates": m.duplicated().sum(),
        "logical_issues": logic_issues,
        "attendance_outliers": len(outliers)
    }
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return m
