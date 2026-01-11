# EPL Match Outcome Predictor - Scientific Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Models](#6-machine-learning-models)
7. [API Infrastructure](#7-api-infrastructure)
8. [Frontend Application](#8-frontend-application)
9. [Database Design](#9-database-design)
10. [Deployment & DevOps](#10-deployment--devops)
11. [Experiment Tracking](#11-experiment-tracking)
12. [Project Structure](#12-project-structure)

---

## 1. Project Overview

The EPL Match Outcome Predictor is a production-grade machine learning system designed to predict English Premier League match outcomes. The system implements a complete ML pipeline from data ingestion through model training to cloud deployment with an interactive web interface.

### Key Capabilities

- **Historical Data Analysis**: Processes 11 seasons (2014-2025) of EPL match data
- **Real-time Predictions**: Provides match outcome probabilities via REST API
- **Interactive Web Interface**: User-friendly frontend for custom predictions
- **Automated Pipeline**: End-to-end orchestration from data download to deployment

### Performance Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | 80.79% |
| Log Loss | 0.4076 |
| Feature Count | 12 |
| Training Data | 9 seasons |

---

## 2. System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Premier League  │    │   matches.csv   │    │  PostgreSQL DB  │          │
│  │      API        │───▶│  standings.csv  │───▶│   team_stats    │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML PIPELINE LAYER                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │    ELO Rating   │    │     Feature     │    │  Model Training │          │
│  │   Calculation   │───▶│   Engineering   │───▶│   (RF/XGB/LGB)  │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                          │                   │
│                                    ┌─────────────────────┘                   │
│                                    ▼                                         │
│                          ┌─────────────────┐                                 │
│                          │  Weights & Biases│                                │
│                          │ Experiment Track │                                │
│                          └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT LAYER                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  Modal Volume   │    │  FastAPI App    │    │   Astro/React   │          │
│  │ (Model Storage) │───▶│ (Inference API) │◀───│   (Web UI)      │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Data Collection**: Premier League API → CSV files → PostgreSQL database
2. **Training Pipeline**: CSV data → Feature engineering → Model training → Model serialization
3. **Inference Flow**: Web UI → FastAPI → Model prediction → Response to user
4. **Experiment Tracking**: Training metrics → Weights & Biases dashboard

---

## 3. Technology Stack

### Backend & Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Primary programming language |
| scikit-learn | 1.5.2 | Random Forest implementation |
| XGBoost | 2.1.4 | Gradient boosting model |
| LightGBM | 4.5.0 | Light gradient boosting model |
| pandas | 2.3.3 | Data manipulation and analysis |
| NumPy | 1.26.4 | Numerical computations |
| FastAPI | 0.115.0 | REST API framework |
| Modal | 0.69.1 | Serverless deployment platform |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| Astro | 5.15.6 | Static site generation framework |
| React | 19.2.0 | UI component library |
| Tailwind CSS | 4.1.17 | Utility-first CSS framework |
| TypeScript | - | Type-safe JavaScript |

### Infrastructure

| Technology | Purpose |
|------------|---------|
| PostgreSQL | Relational database for team statistics |
| Modal.com | Serverless cloud deployment |
| Weights & Biases | Experiment tracking and model versioning |
| Railway | Managed PostgreSQL hosting |

---

## 4. Data Pipeline

### Data Sources

The system collects data from the official Premier League API:

- **Matches Endpoint**: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v1/competitions/8/seasons/{SEASON}/matchweeks/{WEEK}/matches`
- **Standings Endpoint**: `https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{SEASON}/standings`

### Data Collection Process

```
┌──────────────────┐
│  download_data.py │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ For each season (2014-2025):                             │
│   For each matchweek (1-38):                             │
│     - Fetch match data from API                          │
│     - Extract: teams, scores, venue, attendance, date    │
│     - Determine match outcome (H/D/A)                    │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐    ┌──────────────────┐
│   matches.csv    │    │  standings.csv   │
│  (~4000+ rows)   │    │ (season rankings)│
└──────────────────┘    └──────────────────┘
```

### Data Schema (matches.csv)

| Column | Type | Description |
|--------|------|-------------|
| match_id | int | Unique match identifier |
| season | int | Season year (e.g., 2024) |
| matchweek | int | Week number (1-38) |
| kickoff_datetime | datetime | Match start time |
| home_team | str | Home team name |
| away_team | str | Away team name |
| home_score | int | Home team final score |
| away_score | int | Away team final score |
| venue | str | Stadium name |
| attendance | int | Match attendance |
| outcome | str | H (Home), D (Draw), A (Away) |

---

## 5. Feature Engineering

### ELO Rating System

The system implements a chess-style ELO rating to quantify team strength:

**Expected Score Calculation:**
```
E = 1 / (1 + 10^((ELO_away - ELO_home) / 400))
```

**Rating Update:**
```
New_ELO = Old_ELO + K × (Actual - Expected)
```

**Parameters:**
- Initial Rating: 1500 (all teams)
- K-factor: 20 (update magnitude)
- Outcome Values: Win=1.0, Draw=0.5, Loss=0.0

### Feature Set (12 Features)

| Feature | Category | Description |
|---------|----------|-------------|
| `elo_home_pre` | ELO | Home team ELO rating before match |
| `elo_away_pre` | ELO | Away team ELO rating before match |
| `goal_diff_pre` | ELO | ELO difference (home - away) |
| `home_gf_roll` | Form | Home team avg goals scored (5-match rolling) |
| `home_ga_roll` | Form | Home team avg goals conceded (5-match rolling) |
| `home_pts_roll` | Form | Home team avg points (5-match rolling) |
| `away_gf_roll` | Form | Away team avg goals scored (5-match rolling) |
| `away_ga_roll` | Form | Away team avg goals conceded (5-match rolling) |
| `away_pts_roll` | Form | Away team avg points (5-match rolling) |
| `rest_days_home` | Schedule | Days since home team's last match |
| `rest_days_away` | Schedule | Days since away team's last match |
| `rest_days_diff` | Schedule | Rest days difference (home - away) |

### Feature Engineering Pipeline

```python
# Feature preparation flow (src/features.py)
def prepare_data(matches_df):
    # 1. Compute ELO ratings chronologically
    df = compute_elo_ratings(matches_df)

    # 2. Calculate rolling statistics (5-match window)
    df = compute_rolling_stats(df)

    # 3. Calculate rest days between matches
    df = compute_rest_days(df)

    # 4. Create target variable (H_or_D vs A)
    df['target'] = df['outcome'].apply(
        lambda x: 'H_or_D' if x in ['H', 'D'] else 'A'
    )

    return df
```

### Target Variable

The system uses binary classification:
- **H_or_D**: Home Win or Draw (positive class)
- **A**: Away Win (negative class)

**Rationale**: Home advantage is significant in football (~46% home wins vs ~27% away wins). Grouping Home and Draw focuses predictions on "will the away team win?"

---

## 6. Machine Learning Models

### Model Comparison

| Model | Accuracy | Log Loss | Training Time |
|-------|----------|----------|---------------|
| **Random Forest** | 80.79% | 0.4076 | ~15s |
| XGBoost | 80.79% | 0.4448 | ~20s |
| LightGBM | 80.21% | 0.6404 | ~5s |

### Random Forest (Production Model)

**Architecture:**
- 500 decision trees
- Unlimited tree depth
- Balanced class weights
- Bootstrap aggregation

**Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
```

**Selection Rationale:**
- Best probability calibration (lowest log loss)
- Robust to outliers and noisy features
- Captures complex feature interactions
- No feature scaling required

### XGBoost

**Architecture:**
- 500 boosted trees
- Max depth: 6
- Learning rate: 0.05
- Row/column subsampling: 80%

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob'
)
```

### LightGBM

**Architecture:**
- 1000 boosted trees (leaf-wise growth)
- 31 leaves per tree
- Learning rate: 0.05
- Minimum 20 samples per leaf

**Hyperparameters:**
```python
LGBMClassifier(
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.05,
    min_data_in_leaf=20,
    objective='multiclassova'
)
```

### Train-Test Split Strategy

```
Training Data: Seasons 2014-2022 (9 seasons, ~3400 matches)
Testing Data:  Seasons 2023-2025 (2+ seasons, ~600 matches)
```

**Rationale:**
- Temporal split prevents data leakage
- Tests on most recent (harder to predict) matches
- Evaluates model on unseen team dynamics

### Model Evaluation Metrics

| Metric | Description | Value |
|--------|-------------|-------|
| Accuracy | Overall correct predictions | 80.79% |
| Log Loss | Probability calibration quality | 0.4076 |
| Precision | True positives / predicted positives | 0.81 |
| Recall | True positives / actual positives | 0.81 |
| F1-Score | Harmonic mean of precision/recall | 0.81 |
| ROC-AUC | Area under ROC curve | 0.76 |

---

## 7. API Infrastructure

### Deployment Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Modal.com Platform                     │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │   Modal Volume   │    │      FastAPI Application     │  │
│  │  /models/        │◀──▶│  - Load model on startup     │  │
│  │  - rf_model.pkl  │    │  - Handle prediction requests│  │
│  │  - features.pkl  │    │  - Serve team statistics     │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│           ▲                           ▲                     │
│           │                           │                     │
│  ┌────────┴─────────┐    ┌───────────┴────────────────┐   │
│  │  Modal Secrets   │    │     PostgreSQL Database    │   │
│  │  - EPL_API_KEY   │    │     (Railway Hosting)      │   │
│  │  - DATABASE_URL  │    │                            │   │
│  └──────────────────┘    └────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Service metadata and available endpoints |
| `/health` | GET | No | Health check and model status |
| `/teams` | GET | No | List all teams with current statistics |
| `/predict` | POST | Yes | Generate match prediction |
| `/fixtures/next` | GET | No | Get next upcoming EPL match |

### Prediction Endpoint Schema

**Request (POST /predict):**
```json
{
  "home_team": "Manchester City",
  "away_team": "Liverpool",
  "elo_home_pre": 1650,        // optional
  "elo_away_pre": 1620,        // optional
  "home_gf_roll": 2.4,         // optional
  "home_ga_roll": 0.6,         // optional
  "home_pts_roll": 2.8,        // optional
  "away_gf_roll": 2.2,         // optional
  "away_ga_roll": 0.8,         // optional
  "away_pts_roll": 2.6,        // optional
  "rest_days_home": 7,         // optional
  "rest_days_away": 4          // optional
}
```

**Response:**
```json
{
  "prediction": "H_or_D",
  "probability_H_or_D": 0.72,
  "probability_A": 0.28,
  "confidence": 72.0,
  "features_used": {
    "elo_home_pre": 1650,
    "elo_away_pre": 1620,
    "goal_diff_pre": 30,
    ...
  }
}
```

### CORS Configuration

Allowed origins for cross-origin requests:
- `https://dspro1.zayden.ch` (Production)
- `https://epl-match-outcome-predictor.vercel.app` (Backup)
- `http://localhost:4321` (Development)

---

## 8. Frontend Application

### Technology Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Astro 5 Framework                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Static Generation                   │   │
│  │  - Pre-renders pages at build time                  │   │
│  │  - Hydrates React components on client              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────┴───────────────────────────┐     │
│  │               React Components                     │     │
│  │  ┌─────────────────┐   ┌─────────────────────┐   │     │
│  │  │ PredictionForm  │   │ NextMatchPrediction │   │     │
│  │  │ - Team dropdowns│   │ - Auto-fetch fixture│   │     │
│  │  │ - Feature inputs│   │ - Session caching   │   │     │
│  │  │ - Result display│   │ - 5-min TTL         │   │     │
│  │  └─────────────────┘   └─────────────────────┘   │     │
│  └───────────────────────────────────────────────────┘     │
│                          │                                  │
│  ┌───────────────────────┴───────────────────────────┐     │
│  │              Tailwind CSS v4                       │     │
│  │  - Utility-first styling                          │     │
│  │  - Responsive design                              │     │
│  │  - Dark gradient backgrounds                      │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Component Structure

```
web-ui/src/
├── pages/
│   └── index.astro           # Main landing page
├── components/
│   ├── PredictionForm.tsx    # Interactive prediction form
│   └── NextMatchPrediction.tsx # Auto-predict next match
├── layouts/
│   └── Layout.astro          # Base HTML layout
└── styles/
    └── global.css            # Tailwind directives
```

### PredictionForm Component

**Features:**
- Dynamic team selection (fetched from API)
- Optional advanced feature inputs for "what-if" analysis
- Real-time form validation
- Loading states and error handling
- Visual probability bars for results

**State Management:**
```typescript
interface FormState {
  homeTeam: string;
  awayTeam: string;
  advancedOptions: {
    elo_home_pre?: number;
    elo_away_pre?: number;
    // ... other optional features
  };
}
```

### NextMatchPrediction Component

**Features:**
- Automatically fetches next EPL fixture from API
- Makes prediction without user input
- Client-side caching (5-minute TTL in SessionStorage)
- Graceful fallback when no fixtures available

**Caching Strategy:**
```typescript
// Cache structure in SessionStorage
{
  key: 'epl_next_match_prediction',
  data: {
    fixture: { ... },
    prediction: { ... },
    timestamp: 1703419200000
  },
  ttl: 300000 // 5 minutes
}
```

### User Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     Header Section                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          EPL Match Outcome Predictor                │   │
│  │     [Accuracy: 80.79%] [Model: RF] [Features: 12]   │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  Next Match Section                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Upcoming: Team A vs Team B                         │   │
│  │  Prediction: Home/Draw (68% confidence)             │   │
│  │  [████████░░] 68%  vs  [███░░░░░░░] 32%            │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                Custom Prediction Form                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Home Team: [Dropdown ▼]  Away Team: [Dropdown ▼]   │   │
│  │  [▶ Show Advanced Options]                          │   │
│  │  [        Predict Match        ]                    │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Footer                                  │
│  [GitHub Repository Link]                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Database Design

### Schema Definition

```sql
CREATE TABLE team_stats (
    team_name VARCHAR(100) PRIMARY KEY,
    elo_rating FLOAT,
    goals_for_avg FLOAT,
    goals_against_avg FLOAT,
    points_avg FLOAT,
    last_match_date DATE,
    matches_played INT,
    updated_at TIMESTAMP,
    season VARCHAR(20)
);

CREATE INDEX idx_team_stats_elo ON team_stats(elo_rating DESC);
CREATE INDEX idx_team_stats_updated ON team_stats(updated_at);
```

### Data Flow

```
matches.csv ──▶ calculate_team_stats() ──▶ PostgreSQL
                        │
                        ▼
              ┌─────────────────────┐
              │ For each team:      │
              │ - Compute final ELO │
              │ - 5-match rolling   │
              │ - Last match date   │
              │ - Matches played    │
              └─────────────────────┘
```

### Database Operations

| Function | Description |
|----------|-------------|
| `calculate_team_stats_from_matches()` | Process all matches to compute current stats |
| `update_team_stats()` | UPSERT team statistics to PostgreSQL |
| `get_team_stats(team_name)` | Fetch single team's current statistics |
| `list_all_teams()` | Retrieve all teams sorted by ELO |

---

## 10. Deployment & DevOps

### Pipeline Orchestration

```
┌────────────────────────────────────────────────────────────┐
│                     pipeline.py                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Download │─▶│  Train   │─▶│  Update  │─▶│  Deploy  │   │
│  │   Data   │  │  Models  │  │    DB    │  │  Modal   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Pipeline Commands

```bash
# Full pipeline
python pipeline.py

# Skip data download (use existing data)
python pipeline.py --skip-download

# Train specific model only
python pipeline.py --model random_forest

# Run specific steps only
python pipeline.py --steps download train

# Include Modal deployment
python pipeline.py --deploy

# Disable W&B tracking
python pipeline.py --no-wandb
```

### Deployment Process

**1. Model Training:**
```bash
python scripts/train_models.py --model random_forest
# Output: models/random_forest_model.pkl, models/features.pkl
```

**2. Database Update:**
```bash
python scripts/update_database.py
# Updates PostgreSQL with fresh team statistics
```

**3. Modal Deployment:**
```bash
python scripts/deploy_model.py
# Uploads model to Modal Volume

modal deploy modal_api.py
# Deploys/updates FastAPI application
```

**4. Frontend Build:**
```bash
cd web-ui
bun run build
# Output: dist/ folder for static hosting
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string |
| `EPL_API_KEY` | API authentication key |
| `WANDB_API_KEY` | Weights & Biases authentication |

---

## 11. Experiment Tracking

### Weights & Biases Integration

**Dashboard**: `https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor`

### Tracked Artifacts

| Category | Metrics |
|----------|---------|
| **Performance** | Accuracy, Log Loss, Precision, Recall, F1 |
| **Curves** | ROC, Precision-Recall, Calibration |
| **Visualizations** | Confusion Matrix, Feature Importance |
| **Analysis** | Confidence Distribution, Misclassifications |

### Experiment Configuration

```python
wandb.init(
    entity='philip-baumann-hslu',
    project='epl-match-outcome-predictor',
    config={
        'model_type': 'random_forest',
        'n_estimators': 500,
        'features': FEATURE_COLUMNS,
        'train_seasons': TRAIN_SEASONS,
        'test_season': PREDICT_SEASON
    }
)
```

---

## 12. Project Structure

```
epl-match-outcome-predictor/
│
├── src/                           # Core Python modules
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration
│   ├── elo.py                     # ELO rating calculations
│   ├── features.py                # Feature engineering pipeline
│   ├── models.py                  # Model training & evaluation
│   └── database.py                # PostgreSQL operations
│
├── scripts/                       # Pipeline step scripts
│   ├── download_data.py           # Step 1: Data collection
│   ├── train_models.py            # Step 2: Model training
│   ├── update_database.py         # Step 3: Database update
│   └── deploy_model.py            # Step 4: Modal deployment
│
├── data/                          # Data files (generated)
│   ├── matches.csv                # Historical match data
│   └── standings.csv              # Season standings
│
├── models/                        # Trained models (generated)
│   ├── random_forest_model.pkl    # Production model
│   ├── xgboost_model.pkl          # Alternative model
│   ├── lightgbm_model.pkl         # Alternative model
│   └── features.pkl               # Feature list
│
├── web-ui/                        # Astro frontend application
│   ├── src/
│   │   ├── pages/index.astro      # Main page
│   │   ├── components/            # React components
│   │   ├── layouts/               # HTML layouts
│   │   └── styles/                # CSS files
│   ├── package.json               # NPM dependencies
│   └── astro.config.mjs           # Astro configuration
│
├── notebooks/                     # Research notebooks
│   ├── colossal_sound.ipynb       # Data exploration
│   └── predict.ipynb              # Model development
│
├── pipeline.py                    # ML pipeline orchestrator
├── modal_api.py                   # FastAPI deployment code
├── test_api.py                    # API testing suite
├── requirements.txt               # Python dependencies
├── schema.sql                     # Database schema
└── README.md                      # Project overview
```

---

## Appendix A: Configuration Parameters

```python
# src/config.py - Key parameters

# Data Collection
CURRENT_SEASON = 2025
PAST_YEARS = 11
START_MATCHWEEK = 1
END_MATCHWEEK = 38

# ELO System
ELO_K = 20
ELO_HOME_ADV = 60
ELO_BASE = 1500

# Feature Engineering
ROLLING_WINDOW = 5
RANDOM_STATE = 42

# Training
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
PREDICT_SEASON = 2025
TRAIN_TEST_SPLIT_SEASON = 2023

# Features
FEATURE_COLUMNS = [
    'elo_home_pre', 'elo_away_pre', 'goal_diff_pre',
    'home_gf_roll', 'home_ga_roll', 'home_pts_roll',
    'away_gf_roll', 'away_ga_roll', 'away_pts_roll',
    'rest_days_home', 'rest_days_away', 'rest_days_diff'
]
```

---

## Appendix B: API Testing

```bash
# Run API test suite
python test_api.py <API_URL> <API_KEY>

# Example
python test_api.py \
  https://dspro1--epl-predictor-fastapi-app.modal.run \
  $EPL_API_KEY
```

**Test Cases:**
1. Health endpoint verification
2. Simple prediction (teams only)
3. Full features prediction (all 12 features)
4. Underdog scenario testing

---

## Appendix C: Production URLs

| Resource | URL |
|----------|-----|
| API Base | `https://dspro1--epl-predictor-fastapi-app.modal.run` |
| Web UI | `https://dspro1.zayden.ch` |
| API Docs | `{API_URL}/docs` |
| W&B Dashboard | `https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor` |
| GitHub | `https://github.com/DSPRO1/epl-match-outcome-predictor` |
