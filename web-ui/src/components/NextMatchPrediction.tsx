import { useState, useEffect } from 'react';

interface Fixture {
  match_id: number;
  home_team: string;
  away_team: string;
  kickoff: string;
  matchweek: number;
  season: number;
  venue: string | null;
}

interface PredictionResult {
  home_team: string;
  away_team: string;
  prediction: string;
  probabilities: {
    home_or_draw: number;
    away: number;
  };
  confidence: number;
  features_used: Record<string, number>;
  model_used: string;
}

interface CachedData {
  fixture: Fixture;
  predictions: Record<string, PredictionResult>;
  timestamp: number;
}

const CACHE_KEY = 'epl_next_match_prediction';
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

// Radial progress chart component
function RadialChart({
  percent,
  color,
  size = 100,
  strokeWidth = 6,
  label
}: {
  percent: number;
  color: string;
  size?: number;
  strokeWidth?: number;
  label: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--bg-deep)"
            strokeWidth={strokeWidth}
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{
              transition: 'stroke-dashoffset 1s cubic-bezier(0.16, 1, 0.3, 1)',
              filter: `drop-shadow(0 0 8px ${color})`
            }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span
            className="font-mono text-lg font-bold"
            style={{ color }}
          >
            {percent.toFixed(1)}%
          </span>
        </div>
      </div>
      <span className="mt-2 text-xs text-[var(--text-muted)] uppercase tracking-wider">
        {label}
      </span>
    </div>
  );
}

// Model comparison bar
function ModelBar({
  model,
  homeDrawProb,
  awayProb,
  isActive
}: {
  model: string;
  homeDrawProb: number;
  awayProb: number;
  isActive: boolean;
}) {
  const modelLabels: Record<string, string> = {
    'random_forest': 'RF',
    'xgboost': 'XGB',
    'lightgbm': 'LGBM'
  };

  return (
    <div className={`
      p-4 rounded-xl border transition-all duration-300
      ${isActive
        ? 'bg-[var(--bg-elevated)] border-[var(--accent-cyan)]/30'
        : 'bg-[var(--bg-surface)] border-[var(--border-subtle)]'
      }
    `}>
      {/* Model label */}
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-xs text-[var(--text-secondary)] uppercase tracking-wider">
          {modelLabels[model] || model}
        </span>
        {isActive && (
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-cyan)] animate-pulse" />
        )}
      </div>

      {/* Dual bar chart */}
      <div className="space-y-2">
        {/* Home/Draw bar */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[var(--text-muted)] w-6">H/D</span>
          <div className="flex-1 h-2 bg-[var(--bg-deep)] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000 ease-out"
              style={{
                width: `${homeDrawProb * 100}%`,
                background: 'linear-gradient(90deg, var(--accent-cyan-dim), var(--accent-cyan))',
                boxShadow: '0 0 10px var(--accent-cyan)'
              }}
            />
          </div>
          <span className="text-xs font-mono text-[var(--accent-cyan)] w-12 text-right">
            {(homeDrawProb * 100).toFixed(1)}%
          </span>
        </div>

        {/* Away bar */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[var(--text-muted)] w-6">A</span>
          <div className="flex-1 h-2 bg-[var(--bg-deep)] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000 ease-out"
              style={{
                width: `${awayProb * 100}%`,
                background: 'linear-gradient(90deg, var(--accent-magenta-dim), var(--accent-magenta))',
                boxShadow: '0 0 10px var(--accent-magenta)'
              }}
            />
          </div>
          <span className="text-xs font-mono text-[var(--accent-magenta)] w-12 text-right">
            {(awayProb * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default function NextMatchPrediction() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fixture, setFixture] = useState<Fixture | null>(null);
  const [predictions, setPredictions] = useState<Record<string, PredictionResult>>({});

  useEffect(() => {
    loadNextMatchPrediction();
  }, []);

  const loadNextMatchPrediction = async () => {
    try {
      const cached = checkCache();
      if (cached) {
        setFixture(cached.fixture);
        setPredictions(cached.predictions);
        setLoading(false);
        return;
      }

      const apiUrl = import.meta.env.PUBLIC_API_URL || 'https://dspro1--epl-predictor-fastapi-app.modal.run';

      const fixtureResponse = await fetch(`${apiUrl}/fixtures/next`);
      if (!fixtureResponse.ok) {
        if (fixtureResponse.status === 404) {
          setLoading(false);
          return;
        }
        throw new Error('Failed to fetch next fixture');
      }

      const fixtureData: Fixture = await fixtureResponse.json();
      setFixture(fixtureData);

      const apiKey = import.meta.env.PUBLIC_API_KEY;
      if (!apiKey) {
        throw new Error('API key not configured');
      }

      const models = ['random_forest', 'xgboost', 'lightgbm'];

      // Fetch all models in parallel for faster loading
      const predictionPromises = models.map(async (model) => {
        try {
          const response = await fetch(`${apiUrl}/predict`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-API-Key': apiKey,
            },
            body: JSON.stringify({
              home_team: fixtureData.home_team,
              away_team: fixtureData.away_team,
              model: model,
            }),
          });

          if (response.ok) {
            const data: PredictionResult = await response.json();
            return { model, data };
          }
          return null;
        } catch {
          return null;
        }
      });

      const results = await Promise.all(predictionPromises);
      const predictionsData: Record<string, PredictionResult> = {};

      for (const result of results) {
        if (result) {
          predictionsData[result.model] = result.data;
        }
      }

      setPredictions(predictionsData);
      cacheResults(fixtureData, predictionsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load prediction');
      console.error('Error loading next match prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  const checkCache = (): CachedData | null => {
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (!cached) return null;

      const data: CachedData = JSON.parse(cached);
      const now = Date.now();

      if (now - data.timestamp < CACHE_DURATION) {
        return data;
      }

      localStorage.removeItem(CACHE_KEY);
      return null;
    } catch {
      return null;
    }
  };

  const cacheResults = (fixtureData: Fixture, predictionsData: Record<string, PredictionResult>) => {
    try {
      const data: CachedData = {
        fixture: fixtureData,
        predictions: predictionsData,
        timestamp: Date.now(),
      };
      localStorage.setItem(CACHE_KEY, JSON.stringify(data));
    } catch (err) {
      console.warn('Failed to cache results:', err);
    }
  };

  const formatKickoffTime = (kickoff: string) => {
    try {
      const date = new Date(kickoff);
      return new Intl.DateTimeFormat('en-GB', {
        weekday: 'short',
        day: 'numeric',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short',
      }).format(date);
    } catch {
      return kickoff;
    }
  };

  if (loading) {
    return (
      <div className="card-dark card-glow p-8">
        <div className="flex items-center justify-center gap-3">
          <div className="spinner spinner-glow" />
          <span className="text-[var(--text-secondary)] font-mono text-sm">
            Loading predictions<span className="cursor-blink"></span>
          </span>
        </div>
      </div>
    );
  }

  if (error || !fixture || Object.keys(predictions).length === 0) {
    return null;
  }

  const modelOrder = ['random_forest', 'xgboost', 'lightgbm'];
  // Use first available prediction for main display
  const mainPrediction = predictions['random_forest'] || predictions['xgboost'] || predictions['lightgbm'];

  return (
    <div className="card-dark card-glow-cyan overflow-hidden">
      {/* Top gradient bar */}
      <div className="h-1 bg-gradient-to-r from-[var(--accent-cyan)] via-[var(--accent-purple)] to-[var(--accent-magenta)]" />

      <div className="p-6 md:p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--accent-cyan)]/10 border border-[var(--accent-cyan)]/30 rounded-full">
              <span className="w-2 h-2 rounded-full bg-[var(--accent-cyan)] animate-pulse" />
              <span className="font-mono text-xs text-[var(--accent-cyan)] uppercase tracking-wider">
                Next Match
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="font-mono text-xs text-[var(--text-muted)]">
              MW {fixture.matchweek}
            </span>
          </div>
        </div>

        {/* Match Title */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-4 md:gap-8 mb-4">
            <h3 className="text-xl md:text-3xl font-bold text-[var(--text-primary)] flex-1 text-right">
              {fixture.home_team}
            </h3>
            <div className="flex-shrink-0 w-12 h-12 rounded-full border-2 border-[var(--border-subtle)] bg-[var(--bg-elevated)] flex items-center justify-center">
              <span className="font-mono text-xs text-[var(--text-muted)]">VS</span>
            </div>
            <h3 className="text-xl md:text-3xl font-bold text-[var(--text-primary)] flex-1 text-left">
              {fixture.away_team}
            </h3>
          </div>
          <p className="font-mono text-sm text-[var(--text-secondary)]">
            {formatKickoffTime(fixture.kickoff)}
          </p>
          {fixture.venue && (
            <p className="text-xs text-[var(--text-muted)] mt-1">{fixture.venue}</p>
          )}
        </div>

        {/* Main Prediction Display */}
        {mainPrediction && (
          <div className="mb-8">
            {/* Outcome */}
            <div className="text-center mb-6">
              <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Predicted Outcome
              </p>
              <p className={`text-2xl md:text-3xl font-bold ${
                mainPrediction.prediction.includes('Away')
                  ? 'text-[var(--accent-magenta)]'
                  : 'text-[var(--accent-cyan)]'
              }`}>
                {mainPrediction.prediction}
              </p>
            </div>

            {/* Radial Charts */}
            <div className="flex justify-center gap-8 md:gap-16 mb-8">
              <RadialChart
                percent={mainPrediction.probabilities.home_or_draw * 100}
                color="var(--accent-cyan)"
                label="Home/Draw"
                size={110}
                strokeWidth={8}
              />
              <RadialChart
                percent={mainPrediction.confidence * 100}
                color="var(--accent-purple)"
                label="Confidence"
                size={110}
                strokeWidth={8}
              />
              <RadialChart
                percent={mainPrediction.probabilities.away * 100}
                color="var(--accent-magenta)"
                label="Away Win"
                size={110}
                strokeWidth={8}
              />
            </div>
          </div>
        )}

        {/* Model Comparison */}
        <div>
          <h4 className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-wider mb-4">
            Model Comparison
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {modelOrder.map((model) => {
              const pred = predictions[model];
              if (!pred) return null;
              return (
                <ModelBar
                  key={model}
                  model={model}
                  homeDrawProb={pred.probabilities.home_or_draw}
                  awayProb={pred.probabilities.away}
                  isActive={false}
                />
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
