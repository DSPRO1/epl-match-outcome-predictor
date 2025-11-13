"""
Model training and evaluation module.

Provides functions for training Random Forest, XGBoost, and LightGBM models
with Weights & Biases integration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    precision_score, recall_score, f1_score
)
import lightgbm as lgb
from xgboost import XGBClassifier
import wandb
import pickle
from pathlib import Path
from typing import Optional, Tuple

from .config import (
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, LIGHTGBM_PARAMS,
    WANDB_ENTITY, WANDB_PROJECT, ELO_K, ELO_HOME_ADV,
    ROLLING_WINDOW, TRAIN_SEASONS
)


def init_wandb_run(model_name: str, config: dict):
    """
    Initialize a Weights & Biases run.

    Args:
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        wandb.Run object
    """
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{model_name}",
        config=config,
        reinit=True
    )


def log_confusion_matrix_to_wandb(y_test, pred_labels, label_map):
    """Create and log confusion matrix to W&B."""
    cm = confusion_matrix(y_test, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


def log_feature_importance_to_wandb(importances, features, model_name):
    """Create and log feature importance plot to W&B."""
    feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_importance.values, y=feat_importance.index)
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Importance")

    wandb.log({"feature_importance": wandb.Image(plt)})
    plt.close()

    # Also log as a table
    importance_table = wandb.Table(
        columns=["Feature", "Importance"],
        data=[[feat, imp] for feat, imp in zip(feat_importance.index, feat_importance.values)]
    )
    wandb.log({"feature_importance_table": importance_table})


def calculate_metrics(y_test, prob, pred_labels):
    """
    Calculate model performance metrics.

    Args:
        y_test: True labels
        prob: Predicted probabilities
        pred_labels: Predicted labels

    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_test, pred_labels),
        "log_loss": log_loss(y_test, prob),
        "precision": precision_score(y_test, pred_labels, average='weighted'),
        "recall": recall_score(y_test, pred_labels, average='weighted'),
        "f1_score": f1_score(y_test, pred_labels, average='weighted')
    }


def train_random_forest(X_train, y_train, X_test, y_test, features,
                       label_map, use_wandb: bool = True) -> Tuple:
    """
    Train Random Forest model with optional W&B tracking.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        features: List of feature names
        label_map: Mapping of labels to integers
        use_wandb: Whether to use Weights & Biases tracking

    Returns:
        Tuple of (model, probabilities, predicted_labels)
    """
    config = {
        **RANDOM_FOREST_PARAMS,
        "model": "RandomForest",
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    if use_wandb:
        run = init_wandb_run("RandomForest", config)

    print("Training Random Forest...")

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)

    # Predictions
    prob = model.predict_proba(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_test, prob, pred_labels)

    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)
        log_feature_importance_to_wandb(model.feature_importances_, features, "Random Forest")

    # Print results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    if use_wandb:
        run.finish()

    return model, prob, pred_labels


def train_xgboost(X_train, y_train, X_test, y_test, features,
                 label_map, use_wandb: bool = True) -> Tuple:
    """
    Train XGBoost model with optional W&B tracking.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        features: List of feature names
        label_map: Mapping of labels to integers
        use_wandb: Whether to use Weights & Biases tracking

    Returns:
        Tuple of (model, probabilities, predicted_labels)
    """
    config = {
        **XGBOOST_PARAMS,
        "model": "XGBoost",
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    if use_wandb:
        run = init_wandb_run("XGBoost", config)

    print("Training XGBoost...")

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    # Predictions
    prob = model.predict_proba(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_test, prob, pred_labels)

    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)
        log_feature_importance_to_wandb(model.feature_importances_, features, "XGBoost")

    # Print results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    if use_wandb:
        run.finish()

    return model, prob, pred_labels


def train_lightgbm(X_train, y_train, X_test, y_test, features,
                  label_map, use_wandb: bool = True) -> Tuple:
    """
    Train LightGBM model with optional W&B tracking.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        features: List of feature names
        label_map: Mapping of labels to integers
        use_wandb: Whether to use Weights & Biases tracking

    Returns:
        Tuple of (model, probabilities, predicted_labels)
    """
    config = {
        **LIGHTGBM_PARAMS,
        "model": "LightGBM",
        "elo_k": ELO_K,
        "elo_home_adv": ELO_HOME_ADV,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": str(TRAIN_SEASONS),
        "n_features": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    if use_wandb:
        run = init_wandb_run("LightGBM", config)

    print("Training LightGBM...")

    # Extract training params
    num_boost_round = config.pop('num_boost_round')
    params = {k: v for k, v in config.items() if k not in ['model', 'elo_k', 'elo_home_adv', 'rolling_window', 'train_seasons', 'n_features', 'train_samples', 'test_samples']}

    lgb_tr = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, lgb_tr, num_boost_round=num_boost_round)

    # Predictions
    prob = model.predict(X_test)
    pred_labels = np.argmax(prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_test, prob, pred_labels)

    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        log_confusion_matrix_to_wandb(y_test, pred_labels, label_map)
        log_feature_importance_to_wandb(model.feature_importance(importance_type='gain'), features, "LightGBM")

    # Print results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    if use_wandb:
        run.finish()

    return model, prob, pred_labels


def save_model(model, filepath: Path, features: Optional[list] = None):
    """
    Save model to disk.

    Args:
        model: Trained model
        filepath: Path to save the model
        features: Optional list of feature names to save alongside model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")

    # Save features if provided
    if features is not None:
        features_path = filepath.parent / "features.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"✓ Features saved to {features_path}")


def load_model(filepath: Path):
    """
    Load model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
