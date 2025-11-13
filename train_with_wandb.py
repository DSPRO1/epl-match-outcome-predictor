"""
EPL Match Outcome Predictor - Training Script with W&B Integration

This script trains three ML models (Random Forest, LightGBM, XGBoost) and tracks
all experiments, metrics, and artifacts using Weights & Biases.

Usage:
    python train_with_wandb.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from xgboost import XGBClassifier
import wandb
from wandb_config import init_wandb_run

# Import data loader
import data_loader as dl

# Constants
TRAIN_SEASONS = list(range(2019, 2025))
PREDICT_SEASON = 2025
ELO_K = 20
ELO_HOME_ADV = 60
ROLLING_WINDOW = 5
RANDOM_STATE = 42


def init_elo(teams, base=1500):
    """Initialize ELO ratings for all teams."""
    return {t: base for t in teams}


def expected_score(elo_ta, elo_th):
    """Calculate expected score based on ELO ratings."""
    return 1 / (1 + 10 ** ((elo_ta - elo_th) / 400.0))


def compute_elo_ratings(df):
    """
    Compute per-match pre-game ELO ratings for home and away teams.
    Returns two new columns: elo_home_pre, elo_away_pre
    """
    team_col_home = 'home_team'
    team_col_away = 'away_team'
    score_home = 'home_score'
    score_away = 'away_score'
    season_order_col = 'kickoff_datetime'

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

        # Compute outcome
        if row[score_home] > row[score_away]:
            s_h, s_a = 1.0, 0.0
        elif row[score_home] < row[score_away]:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        exp_h = expected_score(elo[ta], elo[th])
        exp_a = 1 - exp_h

        elo[th] = elo[th] + ELO_K * (s_h - exp_h)
        elo[ta] = elo[ta] + ELO_K * (s_a - exp_a)

    # Attach to original index
    df_out = df_sorted.copy()
    df_out['elo_home_pre'] = elo_home_pre
    df_out['elo_away_pre'] = elo_away_pre
    return df_out.sort_index()


def prepare_data(df):
    """
    Prepare features for model training.
    Returns X_train, y_train, X_test, y_test, test_df, label_map, features
    """
    df = df.rename(columns={'winner': 'winner_label', 'outcome': 'outcome_label'}) if 'winner' in df.columns else df

    df = compute_elo_ratings(df)

    df['home_adv'] = 1
    df['goal_diff_pre'] = df['elo_home_pre'] - df['elo_away_pre']

    # Create team-level rows for rolling statistics
    home_rows = df[['match_id', 'kickoff_datetime', 'season', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home_rows.columns = ['match_id', 'kickoff_datetime', 'season', 'team', 'opponent', 'score_for', 'score_against']
    home_rows['is_home'] = 1

    away_rows = df[['match_id', 'kickoff_datetime', 'season', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away_rows.columns = ['match_id', 'kickoff_datetime', 'season', 'team', 'opponent', 'score_for', 'score_against']
    away_rows['is_home'] = 0

    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_rows = team_rows.sort_values(['team', 'kickoff_datetime'])

    # Compute rolling averages
    team_rows['points'] = team_rows.apply(
        lambda r: 3 if r['score_for'] > r['score_against'] else (1 if r['score_for'] == r['score_against'] else 0),
        axis=1
    )
    team_rows[['gf_roll', 'ga_roll', 'pts_roll']] = team_rows.groupby('team')[['score_for', 'score_against', 'points']].rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean().reset_index(level=0, drop=True)[['score_for', 'score_against', 'points']]

    # Merge rolling features back
    team_rows = team_rows.sort_values(['match_id', 'team', 'kickoff_datetime']).drop_duplicates(
        subset=['match_id', 'team'], keep='last'
    )
    home_features = team_rows[team_rows['is_home'] == 1][['match_id', 'gf_roll', 'ga_roll', 'pts_roll']].rename(
        columns=lambda c: f'home_{c}' if c != 'match_id' else c
    )
    away_features = team_rows[team_rows['is_home'] == 0][['match_id', 'gf_roll', 'ga_roll', 'pts_roll']].rename(
        columns=lambda c: f'away_{c}' if c != 'match_id' else c
    )
    df = df.merge(home_features, on='match_id', how='left').merge(away_features, on='match_id', how='left')

    # Compute rest days
    df = df.sort_values('kickoff_datetime').reset_index(drop=True)
    last_kickoff = {}
    rest_days_home = []
    rest_days_away = []

    for _, row in df.iterrows():
        th = row['home_team']
        ta = row['away_team']
        t0 = row['kickoff_datetime']

        if th in last_kickoff:
            delta = (t0 - last_kickoff[th]).total_seconds() / (24 * 3600)
            rest_days_home.append(delta)
        else:
            rest_days_home.append(np.nan)
        last_kickoff[th] = t0

        if ta in last_kickoff:
            delta = (t0 - last_kickoff[ta]).total_seconds() / (24 * 3600)
            rest_days_away.append(delta)
        else:
            rest_days_away.append(np.nan)
        last_kickoff[ta] = t0

    df['rest_days_home'] = rest_days_home
    df['rest_days_away'] = rest_days_away
    df['rest_days_diff'] = df['rest_days_home'] - df['rest_days_away']

    # Create target labels
    label_map = {'H_or_D': 0, 'A': 1}
    df['target'] = df['outcome_label'].map(label_map)

    # Train/test split
    TRAIN_SEASONS_LOCAL = list(range(df['season'].min(), 2023))
    PREDICT_SEASON_LOCAL = list(range(2023, df['season'].max() + 1))

    train_df = df[df['season'].isin(TRAIN_SEASONS_LOCAL)].copy()
    test_df = df[df['season'].isin(PREDICT_SEASON_LOCAL)].copy()

    # Feature selection
    features = [
        'elo_home_pre', 'elo_away_pre', 'goal_diff_pre',
        'home_gf_roll', 'home_ga_roll', 'home_pts_roll',
        'away_gf_roll', 'away_ga_roll', 'away_pts_roll',
        'rest_days_home', 'rest_days_away', 'rest_days_diff'
    ]

    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test, test_df, label_map, features


def log_confusion_matrix_to_wandb(y_test, pred_labels, label_map):
    """Create and log confusion matrix to W&B."""
    cm = confusion_matrix(y_test, pred_labels)

    # Create a nice visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Log to W&B
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


def log_feature_importance_to_wandb(importances, features, model_name):
    """Create and log feature importance plot to W&B."""
    feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_importance.values, y=feat_importance.index)
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Importance")

    # Log to W&B
    wandb.log({"feature_importance": wandb.Image(plt)})
    plt.close()

    # Also log as a table
    importance_table = wandb.Table(
        columns=["Feature", "Importance"],
        data=[[feat, imp] for feat, imp in zip(feat_importance.index, feat_importance.values)]
    )
    wandb.log({"feature_importance_table": importance_table})


def train_random_forest(X_train, y_train, X_test, y_test, test_df, label_map, features):
    """Train Random Forest model with W&B tracking."""

    # Model hyperparameters
    config = {
        "model": "RandomForest",
        "n_estimators": 500,
        "max_depth": None,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # Initialize W&B run
    run = init_wandb_run("RandomForest", config)

    print("Training Random Forest...")

    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
        class_weight=config["class_weight"],
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    prob = rf_model.predict_proba(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_labels)
    logloss = log_loss(y_test, prob)
    precision = precision_score(y_test, pred_labels, average='weighted')
    recall = recall_score(y_test, pred_labels, average='weighted')
    f1 = f1_score(y_test, pred_labels, average='weighted')

    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "log_loss": logloss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log confusion matrix
    log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)

    # Log feature importance
    log_feature_importance_to_wandb(rf_model.feature_importances_, features, "Random Forest")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Finish run
    run.finish()

    return rf_model, prob, pred_labels


def train_lightgbm(X_train, y_train, X_test, y_test, test_df, label_map, features):
    """Train LightGBM model with W&B tracking."""

    # Model hyperparameters
    config = {
        "model": "LightGBM",
        "objective": "multiclassova",
        "num_class": 2,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "num_boost_round": 1000,
        "random_state": RANDOM_STATE,
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # Initialize W&B run
    run = init_wandb_run("LightGBM", config)

    print("Training LightGBM...")

    # Train model
    params = {
        'objective': config["objective"],
        'num_class': config["num_class"],
        'metric': config["metric"],
        'verbosity': -1,
        'seed': config["random_state"],
        'learning_rate': config["learning_rate"],
        'num_leaves': config["num_leaves"],
        'min_data_in_leaf': config["min_data_in_leaf"]
    }

    lgb_tr = lgb.Dataset(X_train, label=y_train)
    gbm = lgb.train(params, lgb_tr, num_boost_round=config["num_boost_round"])

    # Predictions
    prob = gbm.predict(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_labels)
    logloss = log_loss(y_test, prob)
    precision = precision_score(y_test, pred_labels, average='weighted')
    recall = recall_score(y_test, pred_labels, average='weighted')
    f1 = f1_score(y_test, pred_labels, average='weighted')

    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "log_loss": logloss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log confusion matrix
    log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)

    # Log feature importance
    log_feature_importance_to_wandb(gbm.feature_importance(importance_type='gain'), features, "LightGBM")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Finish run
    run.finish()

    return gbm, prob, pred_labels


def train_xgboost(X_train, y_train, X_test, y_test, test_df, label_map, features):
    """Train XGBoost model with W&B tracking."""

    # Model hyperparameters
    config = {
        "model": "XGBoost",
        "n_estimators": 500,
        "num_class": 2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # Initialize W&B run
    run = init_wandb_run("XGBoost", config)

    print("Training XGBoost...")

    # Train model
    xgb_model = XGBClassifier(
        n_estimators=config["n_estimators"],
        num_class=config["num_class"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        objective=config["objective"],
        eval_metric=config["eval_metric"]
    )

    xgb_model.fit(X_train, y_train)

    # Predictions
    prob = xgb_model.predict_proba(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_labels)
    logloss = log_loss(y_test, prob)
    precision = precision_score(y_test, pred_labels, average='weighted')
    recall = recall_score(y_test, pred_labels, average='weighted')
    f1 = f1_score(y_test, pred_labels, average='weighted')

    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "log_loss": logloss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log confusion matrix
    log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)

    # Log feature importance
    log_feature_importance_to_wandb(xgb_model.feature_importances_, features, "XGBoost")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Finish run
    run.finish()

    return xgb_model, prob, pred_labels


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("EPL Match Outcome Predictor - Training with W&B Integration")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    matches, standings = dl.load_data()
    print(f"Loaded {len(matches)} matches")

    # Prepare data
    print("\nPreparing features...")
    X_train, y_train, X_test, y_test, test_df, label_map, features = prepare_data(matches)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {features}")

    # Train all models
    print("\n" + "=" * 80)
    print("Training Models with W&B Tracking")
    print("=" * 80)

    # Random Forest
    print("\n[1/3] Random Forest")
    print("-" * 80)
    rf_model, rf_prob, rf_pred = train_random_forest(
        X_train, y_train, X_test, y_test, test_df, label_map, features
    )

    # LightGBM
    print("\n[2/3] LightGBM")
    print("-" * 80)
    lgb_model, lgb_prob, lgb_pred = train_lightgbm(
        X_train, y_train, X_test, y_test, test_df, label_map, features
    )

    # XGBoost
    print("\n[3/3] XGBoost")
    print("-" * 80)
    xgb_model, xgb_prob, xgb_pred = train_xgboost(
        X_train, y_train, X_test, y_test, test_df, label_map, features
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nView results at: https://wandb.ai/{wandb_config.WANDB_ENTITY}/{wandb_config.WANDB_PROJECT}")


if __name__ == "__main__":
    # Import wandb_config to show URL
    import wandb_config
    main()
