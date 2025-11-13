"""
Configuration module for EPL Match Outcome Predictor.

Centralizes all configuration parameters used across the pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data files
MATCHES_CSV = DATA_DIR / "matches.csv"
STANDINGS_CSV = DATA_DIR / "standings.csv"

# Model files
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest_model.pkl"
XGBOOST_MODEL = MODELS_DIR / "xgboost_model.pkl"
LIGHTGBM_MODEL = MODELS_DIR / "lightgbm_model.pkl"
FEATURES_FILE = MODELS_DIR / "features.pkl"

# Data collection settings
CURRENT_SEASON = 2025
PAST_YEARS = 11
START_MATCHWEEK = 1
END_MATCHWEEK = 38

# ELO rating parameters
ELO_K = 20  # K-factor for ELO updates
ELO_HOME_ADV = 60  # Home advantage (defined but not used in calculation)
ELO_BASE = 1500  # Starting ELO rating for all teams

# Feature engineering parameters
ROLLING_WINDOW = 5  # Number of matches for rolling averages
RANDOM_STATE = 42  # For reproducibility

# Training parameters
TRAIN_SEASONS = list(range(2019, 2025))
PREDICT_SEASON = 2025
TRAIN_TEST_SPLIT_SEASON = 2023  # Train on <2023, test on >=2023

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': None,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
}

XGBOOST_PARAMS = {
    'n_estimators': 500,
    'num_class': 2,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
}

LIGHTGBM_PARAMS = {
    'objective': 'multiclassova',
    'num_class': 2,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'seed': RANDOM_STATE,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'num_boost_round': 1000,
}

# Feature names (used across pipeline)
FEATURE_COLUMNS = [
    'elo_home_pre', 'elo_away_pre', 'goal_diff_pre',
    'home_gf_roll', 'home_ga_roll', 'home_pts_roll',
    'away_gf_roll', 'away_ga_roll', 'away_pts_roll',
    'rest_days_home', 'rest_days_away', 'rest_days_diff'
]

# Target labels
LABEL_MAP = {'H_or_D': 0, 'A': 1}
INVERSE_LABEL_MAP = {0: 'H_or_D', 1: 'A'}

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')

# Weights & Biases configuration
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'dspro1')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'epl-predictor')

# API configuration
MODAL_APP_NAME = "epl-predictor"
API_VERSION = "2.0.0"
