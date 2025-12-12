import pandas as pd
import numpy as np

ELO_K = 20
ELO_HOME_ADV = 60
ROLLING_WINDOW = 5


def init_elo(teams, base=1500):
    return {t: base for t in teams}

def expected_score(elo_ta, elo_th):
    return 1 / (1 + 10 ** ((elo_ta - elo_th) / 400.0))

def compute_elo_ratings(df):
    """
    Compute per-match pre-game ELO ratings for home and away teams.
    Returns two new columns: elo_home_pre, elo_away_pre
    """

    team_col_home='home_team'
    team_col_away='away_team'
    score_home='home_score'
    score_away='away_score'
    season_order_col='kickoff_datetime'

    teams = pd.concat([df[team_col_home], df[team_col_away]]).unique()
    elo = init_elo(teams)
    elo_home_pre = []
    elo_away_pre = []

    df_sorted = df.sort_values(by=season_order_col).reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        th = row[team_col_home]
        ta = row[team_col_away]
        elo_home_pre.append(elo[th])
        elo_away_pre.append(elo[ta])

        # compute outcome
        if row[score_home] > row[score_away]:
            s_h, s_a = 1.0, 0.0
        elif row[score_home] < row[score_away]:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        exp_h = expected_score(elo[ta], elo[th])
        exp_a = 1-exp_h

        elo[th] = elo[th] + ELO_K * (s_h - exp_h)
        elo[ta] = elo[ta] + ELO_K * (s_a - exp_a)

    # attach to original index
    df_out = df_sorted.copy()
    df_out['elo_home_pre'] = elo_home_pre
    df_out['elo_away_pre'] = elo_away_pre
    return df_out.sort_index()  # put back in original order

def create_features(df, label_map):
    df = compute_elo_ratings(df)

    df['elo_diff_pre'] = df['elo_home_pre'] - df['elo_away_pre']

    home_rows = df[['match_id','kickoff_datetime','season','home_team','away_team','home_score','away_score']].copy()
    home_rows.columns = ['match_id','kickoff_datetime','season','team','opponent','score_for','score_against']
    home_rows['is_home'] = 1
    away_rows = df[['match_id','kickoff_datetime','season','away_team','home_team','away_score','home_score']].copy()
    away_rows.columns = ['match_id','kickoff_datetime','season','team','opponent','score_for','score_against']
    away_rows['is_home'] = 0
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)

    team_rows = team_rows.sort_values(['team','kickoff_datetime'])
    #compute rolling avg
    team_rows['points'] = team_rows.apply(lambda r: 3 if r['score_for']>r['score_against'] else (1 if r['score_for']==r['score_against'] else 0), axis=1)
    team_rows[['gf_roll','ga_roll','pts_roll']] = team_rows.groupby('team')[['score_for','score_against','points']].rolling(window=ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)[['score_for','score_against','points']]

    # take last pre-match rolling values for home and away
    team_rows = team_rows.sort_values(['match_id','team','kickoff_datetime']).drop_duplicates(subset=['match_id','team'], keep='last')
    home_features = team_rows[team_rows['is_home']==1][['match_id','gf_roll','ga_roll','pts_roll']].rename(columns=lambda c: f'home_{c}' if c!='match_id' else c)
    away_features = team_rows[team_rows['is_home']==0][['match_id','gf_roll','ga_roll','pts_roll']].rename(columns=lambda c: f'away_{c}' if c!='match_id' else c)
    df = df.merge(home_features, on='match_id', how='left').merge(away_features, on='match_id', how='left')

    df = df.sort_values('kickoff_datetime').reset_index(drop=True)



    last_kickoff = {}
    rest_days_home = []
    rest_days_away = []
    for _, row in df.iterrows():
        th = row['home_team']
        ta = row['away_team']
        t0 = row['kickoff_datetime']

        if th in last_kickoff:
            delta = (t0 - last_kickoff[th]).total_seconds() / (24*3600)
            rest_days_home.append(delta)
        else:
            rest_days_home.append(np.nan)
        last_kickoff[th] = t0

        if ta in last_kickoff:
            delta = (t0 - last_kickoff[ta]).total_seconds() / (24*3600)
            rest_days_away.append(delta)
        else:
            rest_days_away.append(np.nan)
        last_kickoff[ta] = t0
    df['rest_days_home'] = rest_days_home
    df['rest_days_away'] = rest_days_away
    df['rest_days_diff'] = df['rest_days_home'] - df['rest_days_away']

    df['target'] = df['outcome_label'].map(label_map)

    return df

def prepare_data(df):
    df = df.rename(columns={ 'winner': 'winner_label', 'outcome': 'outcome_label' }) if 'winner' in df.columns else df

    train_seasons = list(range(df['season'].min(), 2023))
    test_seasons = list(range(2023,df['season'].max() +1))
    print(train_seasons, test_seasons)
    train_df = df[df['season'].isin(train_seasons)].copy()
    test_df = df[df['season'].isin(test_seasons)].copy()
    label_map = {'H_or_D':0,'A':1}

    train_df = create_features(train_df, label_map)
    test_df = create_features(test_df, label_map)

    features = [
        'elo_home_pre','elo_away_pre','elo_diff_pre',
        'home_gf_roll','home_ga_roll','home_pts_roll',
        'away_gf_roll','away_ga_roll','away_pts_roll',
        'rest_days_home','rest_days_away','rest_days_diff'
    ]

    x_train = train_df[features]
    y_train = train_df['target']

    x_test = test_df[features]
    y_test = test_df['target']

    return x_train, y_train, x_test, y_test, test_df, label_map, features