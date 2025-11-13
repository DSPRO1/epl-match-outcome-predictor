"""
Train ML models for EPL match outcome prediction.

This script:
1. Loads match data from CSV
2. Prepares features using feature engineering pipeline
3. Trains Random Forest, XGBoost, and LightGBM models
4. Logs experiments to Weights & Biases
5. Saves the best model to disk

Usage:
    python scripts/train_models.py [--model MODEL_NAME] [--no-wandb]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import (
    MATCHES_CSV, RANDOM_FOREST_MODEL,
    XGBOOST_MODEL, LIGHTGBM_MODEL, FEATURES_FILE
)
from src.features import prepare_data
from src.models import (
    train_random_forest, train_xgboost, train_lightgbm,
    save_model
)


def main():
    parser = argparse.ArgumentParser(description='Train EPL prediction models')
    parser.add_argument(
        '--model',
        choices=['random_forest', 'xgboost', 'lightgbm', 'all'],
        default='all',
        help='Which model to train (default: all)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases tracking'
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    print("=" * 80)
    print("EPL Match Outcome Predictor - Model Training")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    if not MATCHES_CSV.exists():
        print(f"✗ Data file not found: {MATCHES_CSV}")
        print("  Run 'python scripts/download_data.py' first")
        sys.exit(1)

    matches = pd.read_csv(MATCHES_CSV)
    print(f"Loaded {len(matches)} matches")

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, X_test, y_test, test_df, label_map, features = prepare_data(matches)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {features}")

    # Train models
    print("\n" + "=" * 80)
    print("Training Models")
    if use_wandb:
        print("W&B tracking enabled")
    else:
        print("W&B tracking disabled")
    print("=" * 80)

    models_to_train = []
    if args.model == 'all':
        models_to_train = ['random_forest', 'xgboost', 'lightgbm']
    else:
        models_to_train = [args.model]

    trained_models = {}

    # Random Forest
    if 'random_forest' in models_to_train:
        print("\n[1/3] Random Forest")
        print("-" * 80)
        rf_model, rf_prob, rf_pred = train_random_forest(
            X_train, y_train, X_test, y_test, features, label_map, use_wandb
        )
        trained_models['random_forest'] = rf_model
        save_model(rf_model, RANDOM_FOREST_MODEL, features)

    # XGBoost
    if 'xgboost' in models_to_train:
        print("\n[2/3] XGBoost")
        print("-" * 80)
        xgb_model, xgb_prob, xgb_pred = train_xgboost(
            X_train, y_train, X_test, y_test, features, label_map, use_wandb
        )
        trained_models['xgboost'] = xgb_model
        save_model(xgb_model, XGBOOST_MODEL, features)

    # LightGBM
    if 'lightgbm' in models_to_train:
        print("\n[3/3] LightGBM")
        print("-" * 80)
        lgb_model, lgb_prob, lgb_pred = train_lightgbm(
            X_train, y_train, X_test, y_test, features, label_map, use_wandb
        )
        trained_models['lightgbm'] = lgb_model
        save_model(lgb_model, LIGHTGBM_MODEL, features)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModels saved to: {RANDOM_FOREST_MODEL.parent}")

    if use_wandb:
        from src.config import WANDB_ENTITY, WANDB_PROJECT
        print(f"\nView results at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")


if __name__ == "__main__":
    main()
