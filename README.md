# EPL Match Outcome Predictor ⚽

Professional machine learning pipeline for predicting English Premier League match outcomes using ELO ratings, rolling statistics, and ensemble models with production-ready API deployment.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-Tracking-orange.svg)](https://wandb.ai/dspro1/epl-predictor)
[![Modal](https://img.shields.io/badge/Deployed%20on-Modal-purple.svg)](https://modal.com/)
[![Astro](https://img.shields.io/badge/Web%20UI-Astro-orange.svg)](https://dspro1.zayden.ch)

## 🎯 Overview

This project predicts whether an away team will win an EPL match using tree-based ensemble methods. After initial experiments showed poor performance with multiclass classification (54.6% accuracy) due to the stochastic nature of draws, the problem was reformulated as a binary classification task.

**Selected Production Model:** LightGBM - a gradient boosting ensemble (optimal trade-off between accuracy, precision, and recall)

**Model Comparison:**
- **Random Forest** (69.95% accuracy, 0.5865 log loss, 64.98% precision, 60.45% recall)
- **XGBoost** (71.26% accuracy, **0.5663 log loss**, 61.52% precision, 32.32% recall)
- **LightGBM** (71.05% accuracy, 0.5850 log loss, **66.49% precision**, **63.40% recall) ⭐

All experiments are tracked with **Weights & Biases** for easy comparison and reproducibility.

## 🚀 Quick Start

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/DSPRO1/epl-match-outcome-predictor.git
cd epl-match-outcome-predictor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://..."  # Railway PostgreSQL
export WANDB_API_KEY="..."             # Weights & Biases
```

### Run Full Pipeline

```bash
# Run complete pipeline (download + train + update DB)
python pipeline.py

# Or run individual steps
python pipeline.py --steps download train database

# Train specific model only
python pipeline.py --model random_forest

# Deploy after training
python pipeline.py --deploy
```

### Individual Scripts

```bash
# Step 1: Download data
python scripts/download_data.py

# Step 2: Train models
python scripts/train_models.py --model random_forest

# Step 3: Update database
python scripts/update_database.py

# Step 4: Deploy to Modal
python scripts/deploy_model.py --model random_forest
```

### Start Web UI

```bash
cd web-ui
bun install
bun run dev
# Visit http://localhost:4321
```

### View Results

- **W&B Dashboard:** https://wandb.ai/dspro1/epl-predictor
- **Live Web UI:** https://dspro1.zayden.ch
- **API Endpoint:** https://dspro1--epl-predictor-fastapi-app.modal.run

## 📊 Features

The model uses **14 engineered features** organized into four categories:

**ELO Ratings (3 features):**
- Home team ELO (chess-style rating system)
- Away team ELO (chess-style rating system)
- ELO difference (relative advantage)

**Rest Days (3 features):**
- Home team rest days (recovery time)
- Away team rest days (recovery time)
- Rest day difference

**Rolling Performance (6 features - 5-match window):**
- Home team: goals for, goals against, points average
- Away team: goals for, goals against, points average

**Head-to-Head (2 features):**
- Historical points average for home team vs opponent
- Historical points average for away team vs opponent

## 🏗️ Project Structure

```
epl-match-outcome-predictor/
├── src/                      # Core modules (shared code)
│   ├── config.py            # Centralized configuration
│   ├── elo.py               # ELO rating calculations
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model training & evaluation
│   └── database.py          # Database operations
├── scripts/                  # Pipeline steps
│   ├── download_data.py     # Step 1: Data ingestion
│   ├── train_models.py      # Step 2: Train models
│   ├── update_database.py   # Step 3: Update team stats DB
│   └── deploy_model.py      # Step 4: Deploy to Modal
├── pipeline.py              # Main orchestrator
├── modal_api.py             # FastAPI inference API (deployed on Modal)
├── schema.sql               # PostgreSQL schema for team stats
├── data/                    # Data files (CSV)
├── models/                  # Trained models (PKL)
├── notebooks/               # Research notebooks (archived)
│   ├── colossal_sound.ipynb # Data exploration
│   ├── predict.ipynb        # Initial model development
│   └── data_loader.py       # Data fetching (used by notebooks)
├── web-ui/                  # Astro web interface
├── test_api.py              # API testing utility
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

```
epl-match-outcome-predictor/
├── src/                      # Core modules (shared code)
│   ├── config.py            # Centralized configuration
│   ├── elo.py               # ELO rating calculations
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model training & evaluation
│   └── database.py          # Database operations
├── scripts/                  # Pipeline steps
│   ├── download_data.py     # Step 1: Data ingestion
│   ├── train_models.py      # Step 2: Train models
│   ├── update_database.py   # Step 3: Update team stats DB
│   └── deploy_model.py      # Step 4: Deploy to Modal
├── pipeline.py              # Main orchestrator
├── modal_api.py             # FastAPI inference API (deployed on Modal)
├── schema.sql               # PostgreSQL schema for team stats
├── data/                    # Data files (CSV)
├── models/                  # Trained models (PKL)
├── notebooks/               # Research notebooks (archived)
│   ├── colossal_sound.ipynb # Data exploration
│   ├── predict.ipynb        # Initial model development
│   └── data_loader.py       # Data fetching (used by notebooks)
├── web-ui/                  # Astro web interface
├── test_api.py              # API testing utility
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📈 Model Performance

| Model | Accuracy | Log Loss | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Random Forest | 69.95% | 0.5865 | 64.98% | 60.45% |
| XGBoost | 71.26% | **0.5663** | 61.52% | 32.32% |
| LightGBM | **71.05%** | 0.5850 | **66.49%** | **63.40%** ⭐ |

*Metrics from time-series cross-validation across 11 EPL seasons (2015/16 - 2025/26)*

**Model Selection Rationale:**
- XGBoost achieves the lowest Log Loss but poor Recall (only 32%), failing to detect majority of away wins
- LightGBM selected for production as it offers the optimal trade-off between discrimination and calibration
- Evaluation follows hierarchy: Log Loss → Accuracy → Precision → Recall

## 🔬 Methodology

### Data Pipeline

1. **Data Collection**: Scrape match data from Premier League API (seasons 2015/16 - 2025/26)
2. **ELO Computation**: Calculate pre-match ELO ratings for all teams (K=20, home advantage=60)
3. **Feature Engineering**: Create rolling statistics (5-match window), rest days, and head-to-head records
4. **Time-Series Cross-Validation**: Each fold uses prior seasons for training, current season for testing
5. **Model Training**: Train and compare three tree-based ensemble methods (Random Forest, XGBoost, LightGBM)
6. **Evaluation**: Track metrics, plots, and predictions in Weights & Biases

### Problem Formulation

**Binary Classification:** "Away Team Win" vs "Home/Draw"

Initial multiclass approach (Home/Draw/Away) achieved only 54.6% accuracy due to class imbalance and difficulty predicting draws. The binary formulation significantly improved accuracy to over 70% while providing more reliable probability estimates.

### ELO Rating System

- Initial rating: 1500
- K-factor: 20
- Home advantage: 60 points
- Updated after each match based on result

## 🚀 API Deployment

The API is production-ready and deployed on Modal.com:

**API Base URL:** `https://dspro1--epl-predictor-fastapi-app.modal.run`

### API Endpoints

#### `GET /teams`
List all available teams with current statistics (public endpoint).

```bash
curl https://dspro1--epl-predictor-fastapi-app.modal.run/teams
```

#### `POST /predict`
Predict match outcome (requires `X-API-Key` header).

```bash
curl -X POST "https://dspro1--epl-predictor-fastapi-app.modal.run/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea"
  }'
```

**Response:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "prediction": "Home Win or Draw",
  "probabilities": {
    "home_or_draw": 0.72,
    "away": 0.28
  },
  "confidence": 0.72,
  "features_used": {
    "home_elo": 1761.23,
    "away_elo": 1656.45,
    "goal_diff_pre": 104.78,
    ...
  }
}
```

### Deploy Updates

```bash
# Train and deploy new model
python pipeline.py --deploy

# Or deploy specific model
python scripts/deploy_model.py --model random_forest

# Deploy API changes
modal deploy modal_api.py
```

## 📚 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - API deployment guide for Modal.com
- **[WANDB_GUIDE.md](WANDB_GUIDE.md)** - Complete W&B integration guide
- **[API_AUTH_SETUP.md](API_AUTH_SETUP.md)** - API authentication setup
- **[notebooks/](notebooks/)** - Research notebooks (archived)

## 🔗 Data Sources

**Premier League Official API:**
- Match data: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{SEASON}/matchweeks/{WEEK}/matches`
- Standings: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{SEASON}/standings`

**Reference:**
- [EPL Table](https://www.premierleague.com/en/tables?competition=8&season=2025&round=L_1&matchweek=-1&ha=-1)
- [EA FC25 Dataset](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats) (future work)

