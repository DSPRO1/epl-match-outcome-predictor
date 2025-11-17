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
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve
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
        settings=wandb.Settings(init_timeout=120)
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


def log_roc_curve_to_wandb(y_test, prob):
    """Create and log ROC curve with AUC to W&B."""
    # For binary classification, use probabilities of the positive class (Away wins)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    wandb.log({
        "roc_curve": wandb.Image(plt),
        "roc_auc": roc_auc
    })
    plt.close()

    return roc_auc


def log_precision_recall_curve_to_wandb(y_test, prob):
    """Create and log Precision-Recall curve to W&B."""
    precision, recall, thresholds = precision_recall_curve(y_test, prob[:, 1])
    avg_precision = average_precision_score(y_test, prob[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    wandb.log({
        "precision_recall_curve": wandb.Image(plt),
        "average_precision": avg_precision
    })
    plt.close()

    return avg_precision


def log_calibration_curve_to_wandb(y_test, prob):
    """Create and log calibration curve to W&B."""
    prob_true, prob_pred = calibration_curve(y_test, prob[:, 1], n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    wandb.log({"calibration_curve": wandb.Image(plt)})
    plt.close()


def log_per_class_metrics_to_wandb(y_test, pred_labels, label_map):
    """Log per-class metrics breakdown."""
    # Get classification report as dict
    target_names = list(label_map.keys())
    report = classification_report(y_test, pred_labels, target_names=target_names, output_dict=True)

    # Create a table for per-class metrics
    class_data = []
    for class_name in target_names:
        if class_name in report:
            class_data.append([
                class_name,
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score'],
                report[class_name]['support']
            ])

    table = wandb.Table(
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
        data=class_data
    )
    wandb.log({"per_class_metrics": table})

    # Log individual class metrics
    for class_name in target_names:
        if class_name in report:
            wandb.log({
                f"class_{class_name}_precision": report[class_name]['precision'],
                f"class_{class_name}_recall": report[class_name]['recall'],
                f"class_{class_name}_f1": report[class_name]['f1-score']
            })


def log_prediction_confidence_to_wandb(prob, pred_labels):
    """Analyze and log prediction confidence distribution."""
    # Get max probability for each prediction (confidence)
    confidences = np.max(prob, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean: {confidences.mean():.3f}')
    plt.legend()
    plt.grid(alpha=0.3)

    wandb.log({
        "confidence_distribution": wandb.Image(plt),
        "mean_confidence": confidences.mean(),
        "median_confidence": np.median(confidences),
        "low_confidence_pct": (confidences < 0.6).sum() / len(confidences) * 100
    })
    plt.close()


def log_misclassification_analysis_to_wandb(y_test, pred_labels, prob, X_test, features):
    """Analyze misclassified predictions."""
    # Find misclassified samples
    misclassified_mask = y_test != pred_labels
    n_misclassified = misclassified_mask.sum()

    # Get confidence of misclassified predictions
    misclassified_confidences = np.max(prob[misclassified_mask], axis=1)

    # Create a summary
    misclassification_data = {
        "total_misclassified": n_misclassified,
        "misclassification_rate": n_misclassified / len(y_test) * 100,
        "avg_misclassified_confidence": misclassified_confidences.mean() if n_misclassified > 0 else 0
    }

    wandb.log(misclassification_data)

    # Plot confidence of correct vs incorrect predictions
    plt.figure(figsize=(10, 6))
    correct_confidences = np.max(prob[~misclassified_mask], axis=1)

    plt.hist(correct_confidences, bins=20, alpha=0.5, label='Correct', color='green')
    plt.hist(misclassified_confidences, bins=20, alpha=0.5, label='Incorrect', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(alpha=0.3)

    wandb.log({"correct_vs_incorrect_confidence": wandb.Image(plt)})
    plt.close()


def log_sample_predictions_to_wandb(y_test, pred_labels, prob, X_test, features, n_samples=20):
    """Log a sample of predictions with their features and probabilities."""
    # Sample random indices
    sample_indices = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)

    # Create table data
    table_data = []
    for idx in sample_indices:
        row = [
            int(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else int(y_test[idx]),
            int(pred_labels[idx]),
            float(prob[idx, 0]),
            float(prob[idx, 1]),
            'Correct' if y_test.iloc[idx] == pred_labels[idx] else 'Wrong'
        ]
        # Add key features
        if hasattr(X_test, 'iloc'):
            row.extend([float(X_test.iloc[idx][feat]) for feat in features[:5]])
        else:
            row.extend([float(X_test[idx, i]) for i in range(min(5, len(features)))])
        table_data.append(row)

    columns = ["Actual", "Predicted", "Prob_H_or_D", "Prob_A", "Result"] + features[:5]
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"sample_predictions": table})


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

        # Enhanced W&B logging
        log_roc_curve_to_wandb(y_test, prob)
        log_precision_recall_curve_to_wandb(y_test, prob)
        log_calibration_curve_to_wandb(y_test, prob)
        log_per_class_metrics_to_wandb(y_test, pred_labels, label_map)
        log_prediction_confidence_to_wandb(prob, pred_labels)
        log_misclassification_analysis_to_wandb(y_test, pred_labels, prob, X_test, features)
        log_sample_predictions_to_wandb(y_test, pred_labels, prob, X_test, features)

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

        # Enhanced W&B logging
        log_roc_curve_to_wandb(y_test, prob)
        log_precision_recall_curve_to_wandb(y_test, prob)
        log_calibration_curve_to_wandb(y_test, prob)
        log_per_class_metrics_to_wandb(y_test, pred_labels, label_map)
        log_prediction_confidence_to_wandb(prob, pred_labels)
        log_misclassification_analysis_to_wandb(y_test, pred_labels, prob, X_test, features)
        log_sample_predictions_to_wandb(y_test, pred_labels, prob, X_test, features)

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

        # Enhanced W&B logging
        log_roc_curve_to_wandb(y_test, prob)
        log_precision_recall_curve_to_wandb(y_test, prob)
        log_calibration_curve_to_wandb(y_test, prob)
        log_per_class_metrics_to_wandb(y_test, pred_labels, label_map)
        log_prediction_confidence_to_wandb(prob, pred_labels)
        log_misclassification_analysis_to_wandb(y_test, pred_labels, prob, X_test, features)
        log_sample_predictions_to_wandb(y_test, pred_labels, prob, X_test, features)

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
