"""
Feature engineering module.

Handles all feature creation for the EPL match outcome prediction model.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .config import (
    ROLLING_WINDOW, TRAIN_TEST_SPLIT_SEASON,
    LABEL_MAP, FEATURE_COLUMNS
)
from .elo import compute_elo_ratings


def create_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling statistics for goals for, goals against, and points.

    Args:
        df: DataFrame with match data

    Returns:
        DataFrame with rolling statistics added
    """
    # Create team-level rows (each match generates 2 rows - one per team)
    home_rows = df[[
        'match_id', 'kickoff_datetime', 'season',
        'home_team', 'away_team', 'home_score', 'away_score'
    ]].copy()
    home_rows.columns = [
        'match_id', 'kickoff_datetime', 'season',
        'team', 'opponent', 'score_for', 'score_against'
    ]
    home_rows['is_home'] = 1

    away_rows = df[[
        'match_id', 'kickoff_datetime', 'season',
        'away_team', 'home_team', 'away_score', 'home_score'
    ]].copy()
    away_rows.columns = [
        'match_id', 'kickoff_datetime', 'season',
        'team', 'opponent', 'score_for', 'score_against'
    ]
    away_rows['is_home'] = 0

    # Combine and sort by team and time
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_rows = team_rows.sort_values(['team', 'kickoff_datetime'])

    # Calculate points
    def calc_points(row):
        if row['score_for'] > row['score_against']:
            return 3
        elif row['score_for'] == row['score_against']:
            return 1
        else:
            return 0

    team_rows['points'] = team_rows.apply(calc_points, axis=1)

    # Calculate rolling averages
    team_rows[['gf_roll', 'ga_roll', 'pts_roll']] = (
        team_rows.groupby('team')[['score_for', 'score_against', 'points']]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        [['score_for', 'score_against', 'points']]
    )

    # Take last pre-match rolling values for home and away
    team_rows = team_rows.sort_values(
        ['match_id', 'team', 'kickoff_datetime']
    ).drop_duplicates(subset=['match_id', 'team'], keep='last')

    # Split into home and away features
    home_features = team_rows[team_rows['is_home'] == 1][
        ['match_id', 'gf_roll', 'ga_roll', 'pts_roll']
    ].rename(columns=lambda c: f'home_{c}' if c != 'match_id' else c)

    away_features = team_rows[team_rows['is_home'] == 0][
        ['match_id', 'gf_roll', 'ga_roll', 'pts_roll']
    ].rename(columns=lambda c: f'away_{c}' if c != 'match_id' else c)

    # Merge back to original dataframe
    df = df.merge(home_features, on='match_id', how='left')
    df = df.merge(away_features, on='match_id', how='left')

    return df


def create_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rest days between matches for each team.

    Args:
        df: DataFrame with match data

    Returns:
        DataFrame with rest_days_home, rest_days_away, rest_days_diff columns
    """
    df = df.sort_values('kickoff_datetime').reset_index(drop=True)

    last_kickoff = {}
    rest_days_home = []
    rest_days_away = []

    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        kickoff_time = row['kickoff_datetime']

        # Calculate home team rest days
        if home_team in last_kickoff:
            delta = (kickoff_time - last_kickoff[home_team]).total_seconds() / (24 * 3600)
            rest_days_home.append(delta)
        else:
            rest_days_home.append(np.nan)
        last_kickoff[home_team] = kickoff_time

        # Calculate away team rest days
        if away_team in last_kickoff:
            delta = (kickoff_time - last_kickoff[away_team]).total_seconds() / (24 * 3600)
            rest_days_away.append(delta)
        else:
            rest_days_away.append(np.nan)
        last_kickoff[away_team] = kickoff_time

    df['rest_days_home'] = rest_days_home
    df['rest_days_away'] = rest_days_away
    df['rest_days_diff'] = df['rest_days_home'] - df['rest_days_away']

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Args:
        df: Raw match data

    Returns:
        DataFrame with all features ready for model training
    """
    # Rename target columns if needed
    if 'winner' in df.columns:
        df = df.rename(columns={'winner': 'winner_label', 'outcome': 'outcome_label'})

    # Compute ELO ratings
    df = compute_elo_ratings(df)

    # Add ELO difference
    df['goal_diff_pre'] = df['elo_home_pre'] - df['elo_away_pre']

    # Add home advantage indicator
    df['home_adv'] = 1

    # Create rolling statistics
    df = create_rolling_statistics(df)

    # Create rest days features
    df = create_rest_days(df)

    # Convert 3-way outcome to 2-way (H_or_D vs A) for binary classification
    df['outcome_binary'] = df['outcome_label'].apply(lambda x: 'H_or_D' if x in ['H', 'D'] else x)

    # Create target variable
    df['target'] = df['outcome_binary'].map(LABEL_MAP)

    return df


def split_train_test(df: pd.DataFrame,
                     split_season: int = TRAIN_TEST_SPLIT_SEASON
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets based on season.

    Args:
        df: Prepared dataframe with features
        split_season: Season to split on (train on <split_season, test on >=split_season)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Split by season
    train_df = df[df['season'] < split_season].copy()
    test_df = df[df['season'] >= split_season].copy()

    # Extract features and targets
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['target']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test


def prepare_data(df: pd.DataFrame) -> Tuple:
    """
    Complete data preparation pipeline.

    Args:
        df: Raw match data

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, test_df, label_map, features)
    """
    # Prepare all features
    df = prepare_features(df)

    # Split into train/test
    X_train, y_train, X_test, y_test = split_train_test(df)

    # Get test dataframe for later analysis
    test_df = df[df['season'] >= TRAIN_TEST_SPLIT_SEASON].copy()

    return X_train, y_train, X_test, y_test, test_df, LABEL_MAP, FEATURE_COLUMNS
