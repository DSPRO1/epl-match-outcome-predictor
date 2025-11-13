# Weights & Biases Integration Guide

This guide explains how to use Weights & Biases (W&B) experiment tracking in the EPL Match Outcome Predictor project.

## Overview

W&B integration allows you to:
- 📊 Track all model experiments in one place
- 📈 Compare metrics across different models (Random Forest, LightGBM, XGBoost)
- 🎨 Visualize feature importance and confusion matrices
- 🔄 Reproduce experiments with tracked hyperparameters
- 📝 Monitor training progress in real-time

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `wandb` - W&B SDK
- `scikit-learn` - Random Forest
- `xgboost` - XGBoost
- `lightgbm` - LightGBM
- `matplotlib`, `seaborn` - Visualization
- And all other dependencies

### 2. Login to W&B

```bash
wandb login
```

When prompted, paste your API key:
```
6a59e9eeb33878d0ad8828b5b134749f96ff1f82
```

Alternatively, set it in your environment:
```bash
export WANDB_API_KEY="6a59e9eeb33878d0ad8828b5b134749f96ff1f82"
```

### 3. Run Training with W&B Tracking

```bash
python train_with_wandb.py
```

This will:
1. Load EPL match data from the Premier League API
2. Compute ELO ratings and features
3. Train 3 models: Random Forest, LightGBM, XGBoost
4. Log all metrics, plots, and hyperparameters to W&B
5. Create separate runs for each model

### 4. View Results

After training, view your results at:
```
https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor
```

## What Gets Tracked?

### Hyperparameters (wandb.config)

For each model, we track:

**Common Parameters:**
- `model` - Model name (RandomForest/LightGBM/XGBoost)
- `elo_k` - ELO K-factor (20)
- `elo_home_adv` - Home advantage (60)
- `rolling_window` - Rolling statistics window (5 matches)
- `train_seasons` - Seasons used for training
- `n_features` - Number of features (12)
- `train_samples` - Training set size
- `test_samples` - Test set size

**Model-Specific:**

*Random Forest:*
- `n_estimators` - Number of trees (500)
- `max_depth` - Maximum tree depth (None)
- `class_weight` - Class balancing (balanced)

*LightGBM:*
- `objective` - Loss function (multiclassova)
- `learning_rate` - Learning rate (0.05)
- `num_leaves` - Max leaves per tree (31)
- `num_boost_round` - Boosting iterations (1000)

*XGBoost:*
- `n_estimators` - Number of boosting rounds (500)
- `max_depth` - Maximum tree depth (6)
- `learning_rate` - Learning rate (0.05)
- `subsample` - Row sampling ratio (0.8)
- `colsample_bytree` - Column sampling ratio (0.8)

### Metrics (wandb.log)

For each model run:
- `accuracy` - Classification accuracy
- `log_loss` - Logarithmic loss (lower is better)
- `precision` - Weighted precision
- `recall` - Weighted recall
- `f1_score` - Weighted F1 score

### Visualizations

1. **Confusion Matrix** (`confusion_matrix`)
   - Shows True Positives, False Positives, True Negatives, False Negatives
   - Labels: H_or_D (Home or Draw) vs A (Away)

2. **Feature Importance** (`feature_importance`)
   - Bar chart showing which features matter most
   - Logged as both image and interactive table
   - Key features: ELO ratings, rolling form, rest days

### Features Used

The model uses 12 engineered features:

1. **ELO Ratings (3):**
   - `elo_home_pre` - Home team ELO before match
   - `elo_away_pre` - Away team ELO before match
   - `goal_diff_pre` - ELO difference

2. **Rolling Form (6):**
   - `home_gf_roll` - Home goals for (5-match avg)
   - `home_ga_roll` - Home goals against (5-match avg)
   - `home_pts_roll` - Home points (5-match avg)
   - `away_gf_roll` - Away goals for (5-match avg)
   - `away_ga_roll` - Away goals against (5-match avg)
   - `away_pts_roll` - Away points (5-match avg)

3. **Rest Days (3):**
   - `rest_days_home` - Days since home team's last match
   - `rest_days_away` - Days since away team's last match
   - `rest_days_diff` - Rest day difference

## Project Configuration

The W&B project is configured in `wandb_config.py`:

```python
WANDB_ENTITY = "philip-baumann-hslu"
WANDB_PROJECT = "epl-match-outcome-predictor"
```

Each model creates a separate run:
- `RandomForest-run`
- `LightGBM-run`
- `XGBoost-run`

## Comparing Models

In the W&B dashboard, you can:

1. **Compare Metrics Side-by-Side**
   - Select all 3 runs
   - View accuracy, log loss, F1 scores in parallel
   - Identify the best performing model

2. **Analyze Feature Importance**
   - Compare which features each model prioritizes
   - Understand model behavior differences

3. **Review Confusion Matrices**
   - See where each model makes mistakes
   - Identify systematic biases

## Example Results

Based on historical performance:

| Model | Accuracy | Log Loss | Notes |
|-------|----------|----------|-------|
| Random Forest | ~80.79% | ~0.4076 | Best calibrated probabilities |
| XGBoost | ~80.79% | ~0.4448 | Close second |
| LightGBM | ~80.21% | ~0.6404 | Fastest training |

## Advanced Usage

### Custom Configuration

Edit `wandb_config.py` to change:
- Project name
- Entity (team name)
- Tags, notes, or run naming

### Manual Run Creation

```python
import wandb
from wandb_config import WANDB_ENTITY, WANDB_PROJECT, setup_wandb

setup_wandb()

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="my-custom-run",
    config={"custom_param": "value"}
)

# Your training code here
wandb.log({"metric": value})

run.finish()
```

### Offline Mode

If you need to train without internet:

```bash
export WANDB_MODE=offline
python train_with_wandb.py
```

Sync later:
```bash
wandb sync wandb/offline-run-*
```

## Troubleshooting

### "API key not found"

Make sure you've logged in:
```bash
wandb login
```

Or set the environment variable:
```bash
export WANDB_API_KEY="6a59e9eeb33878d0ad8828b5b134749f96ff1f82"
```

### "Module not found" errors

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Runs not appearing in dashboard

- Check internet connection
- Verify entity/project names in `wandb_config.py`
- Look for error messages in console output
- Try: `wandb sync --sync-all`

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Quickstart](https://docs.wandb.ai/quickstart)
- [Model Tracking Guide](https://docs.wandb.ai/guides/track)
- [Your Dashboard](https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor)

## Architecture

```
Data Pipeline:
Premier League API
    ↓
data_loader.py
    ↓
ELO Computation + Feature Engineering
    ↓
Train/Test Split
    ↓
┌─────────────┬──────────────┬──────────────┐
│ Random      │  LightGBM    │  XGBoost     │
│ Forest      │              │              │
└─────────────┴──────────────┴──────────────┘
         │           │            │
         └───────────┴────────────┘
                     ↓
              W&B Tracking:
              - Metrics
              - Plots
              - Hyperparameters
              - Artifacts
```

## Next Steps

1. ✅ Run `python train_with_wandb.py` to create your first tracked experiments
2. 📊 Explore the W&B dashboard to compare models
3. 🎯 Tune hyperparameters and create new runs
4. 📈 Track model improvements over time
5. 🤝 Share results with your team using W&B reports
