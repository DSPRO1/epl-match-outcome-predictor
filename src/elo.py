"""
ELO rating calculation module.

Implements the ELO rating system for team strength estimation.
Uses the same calculation as predict.ipynb - home advantage is defined
but not used in the expected score calculation.
"""

import pandas as pd
from typing import Dict
from .config import ELO_BASE, ELO_K


def init_elo(teams, base: float = ELO_BASE) -> Dict[str, float]:
    """
    Initialize ELO ratings for all teams.

    Args:
        teams: Iterable of team names
        base: Starting ELO rating (default: 1500)

    Returns:
        Dictionary mapping team names to initial ELO ratings
    """
    return {team: base for team in teams}


def expected_score(elo_ta: float, elo_th: float) -> float:
    """
    Calculate expected score based on ELO ratings.

    Args:
        elo_ta: Away team ELO rating
        elo_th: Home team ELO rating

    Returns:
        Expected score for home team (between 0 and 1)
    """
    return 1 / (1 + 10 ** ((elo_ta - elo_th) / 400.0))


def compute_elo_update(home_score: float, away_score: float,
                       home_elo: float, away_elo: float,
                       k: float = ELO_K) -> tuple[float, float]:
    """
    Compute updated ELO ratings after a match.

    Args:
        home_score: Home team score
        away_score: Away team score
        home_elo: Home team current ELO rating
        away_elo: Away team current ELO rating
        k: K-factor for ELO updates (default from config)

    Returns:
        Tuple of (new_home_elo, new_away_elo)
    """
    # Determine actual match outcome
    if home_score > away_score:
        home_actual, away_actual = 1.0, 0.0
    elif home_score < away_score:
        home_actual, away_actual = 0.0, 1.0
    else:
        home_actual, away_actual = 0.5, 0.5

    # Calculate expected scores
    home_expected = expected_score(away_elo, home_elo)
    away_expected = 1 - home_expected

    # Update ELO ratings
    new_home_elo = home_elo + k * (home_actual - home_expected)
    new_away_elo = away_elo + k * (away_actual - away_expected)

    return new_home_elo, new_away_elo


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-match ELO ratings for all matches in a dataset.

    Processes matches chronologically and calculates ELO ratings before
    each match, then updates them based on the match outcome.

    Args:
        df: DataFrame with columns: home_team, away_team, home_score,
            away_score, kickoff_datetime

    Returns:
        DataFrame with additional columns: elo_home_pre, elo_away_pre
    """
    # Get unique teams
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo = init_elo(teams)

    # Storage for pre-match ELO ratings
    elo_home_pre = []
    elo_away_pre = []

    # Sort by kickoff time to process chronologically
    df_sorted = df.sort_values(by='kickoff_datetime').reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        # Record pre-match ELO
        elo_home_pre.append(elo[home_team])
        elo_away_pre.append(elo[away_team])

        # Update ELO based on match outcome
        new_home_elo, new_away_elo = compute_elo_update(
            row['home_score'], row['away_score'],
            elo[home_team], elo[away_team]
        )

        elo[home_team] = new_home_elo
        elo[away_team] = new_away_elo

    # Add ELO columns to dataframe
    df_out = df_sorted.copy()
    df_out['elo_home_pre'] = elo_home_pre
    df_out['elo_away_pre'] = elo_away_pre

    # Return in original order
    return df_out.sort_index()
