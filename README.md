# EPL Match Outcome Predictor ⚽

Machine Learning models to predict English Premier League match outcomes using historical data, ELO ratings, and team form statistics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-Tracking-orange.svg)](https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project predicts EPL match outcomes (Home/Draw vs Away) using three machine learning models:
- **Random Forest** (~80.79% accuracy, 0.41 log loss)
- **XGBoost** (~80.79% accuracy, 0.44 log loss)
- **LightGBM** (~80.21% accuracy, 0.64 log loss)

All experiments are tracked with **Weights & Biases** for easy comparison and reproducibility.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DSPRO1/epl-match-outcome-predictor.git
cd epl-match-outcome-predictor

# Install dependencies
pip install -r requirements.txt

# Login to W&B
wandb login
# Paste API key when prompted: 6a59e9eeb33878d0ad8828b5b134749f96ff1f82
```

### Run Training

```bash
python train_with_wandb.py
```

This will:
1. Load 11 seasons of EPL data (2014-2025)
2. Compute ELO ratings and 12 engineered features
3. Train all 3 models
4. Track everything in W&B

### View Results

Visit your W&B dashboard:
```
https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor
```

## 📊 Features

The model uses **12 engineered features**:

**ELO Ratings (3):**
- Home/Away team ELO (chess-style rating system)
- ELO difference

**Rolling Form (6):**
- Goals for/against (5-match averages)
- Points (5-match averages)

**Rest Days (3):**
- Days since last match for each team
- Rest day difference

## 🏗️ Project Structure

```
epl-match-outcome-predictor/
├── train_with_wandb.py      # Main training script with W&B tracking
├── wandb_config.py           # W&B configuration (API key, project)
├── data_loader.py            # Premier League API data fetching
├── predict.ipynb             # Original notebook (legacy)
├── requirements.txt          # All dependencies
├── WANDB_GUIDE.md           # Comprehensive W&B documentation
└── README.md                # This file
```

## 📈 Model Performance

| Model | Accuracy | Log Loss | Training Time |
|-------|----------|----------|---------------|
| Random Forest | 80.79% | 0.4076 | ~15s |
| XGBoost | 80.79% | 0.4448 | ~20s |
| LightGBM | 80.21% | 0.6404 | ~5s |

*Metrics from test set (2023-2025 seasons)*

## 🔬 Methodology

### Data Pipeline

1. **Data Collection**: Scrape match data from Premier League API
2. **ELO Computation**: Calculate pre-match ELO ratings for all teams
3. **Feature Engineering**: Create rolling statistics and rest days
4. **Train/Test Split**: Train on 2014-2022, test on 2023-2025
5. **Model Training**: Train 3 models with different approaches
6. **Evaluation**: Track metrics, plots, and predictions in W&B

### ELO Rating System

- Initial rating: 1500
- K-factor: 20
- Home advantage: 60 points
- Updated after each match based on result

## 📚 Documentation

- **[WANDB_GUIDE.md](WANDB_GUIDE.md)** - Complete W&B integration guide
- **[predict.ipynb](predict.ipynb)** - Jupyter notebook with exploratory analysis
- **[data_loader.py](data_loader.py)** - API documentation and data validation

## 🔗 Data Sources

**Premier League Official API:**
- Match data: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{SEASON}/matchweeks/{WEEK}/matches`
- Standings: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{SEASON}/standings`

**Reference:**
- [EPL Table](https://www.premierleague.com/en/tables?competition=8&season=2025&round=L_1&matchweek=-1&ha=-1)
- [EA FC25 Dataset](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats) (future work)

