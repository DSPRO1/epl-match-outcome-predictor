import requests
import pandas as pd
from datetime import date

def load_data():
    """
    Load EPL match outcomes from Premier League API.

    """

    season = 2025
    past_years = 11
    start_matchweek = 1
    end_matchweek = 38

    all_matches = []

    for year in range(past_years):
        if season - year == season:
            season_2025_start_date = date(2025, 8, 15)
            actual_date = date.today()
            end_matchweek = round((actual_date - season_2025_start_date).days / 7) - 1
        else:
            end_matchweek = 38

        for matchweek in range(start_matchweek, end_matchweek + 1):
            matches_url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{season-year}/matchweeks/{matchweek}/matches"

            try:
                response = requests.get(matches_url, timeout=10)

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
                            'season': season-year,
                            'kickoff': match.get('kickoff'),

                            # Home team info
                            'home_team': home_team.get('name'),
                            'home_team_id': home_team.get('id'),
                            'home_score': home_team.get('score'),

                            # Away team info
                            'away_team': away_team.get('name'),
                            'away_team_id': away_team.get('id'),
                            'away_score': away_team.get('score'),
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
        print(matches_df[['home_team', 'away_team', 'home_score', 'away_score', 'outcome']].head(5))

    matches_df = adjust_values(matches_df)

    return matches_df

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
    print("Goals negative:", ((m['away_score'] < 0) | (m['home_score'] < 0)).sum())

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

    summary = {
        "rows": len(m),
        "columns": len(m.columns),
        "missing_values": m.isna().sum().sum(),
        "duplicates": m.duplicated().sum(),
        "logical_issues": logic_issues,
    }
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return m
