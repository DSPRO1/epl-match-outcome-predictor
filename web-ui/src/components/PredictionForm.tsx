import { useState, useEffect } from 'react';

interface MatchInput {
  home_team: string;
  away_team: string;
  model?: string;
  home_elo?: number;
  away_elo?: number;
  home_gf_roll?: number;
  home_ga_roll?: number;
  home_pts_roll?: number;
  away_gf_roll?: number;
  away_ga_roll?: number;
  away_pts_roll?: number;
  rest_days_home?: number;
  rest_days_away?: number;
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

interface Team {
  team_name: string;
  elo_rating: number;
  goals_for_avg: number;
  goals_against_avg: number;
  points_avg: number;
  last_match_date: string;
  matches_played: number;
}

// Horizontal bar chart for probabilities
function ProbabilityBar({
  label,
  value,
  type
}: {
  label: string;
  value: number;
  type: 'cyan' | 'magenta';
}) {
  const isCyan = type === 'cyan';

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-[var(--text-muted)] uppercase tracking-wider">
          {label}
        </span>
        <span className={`font-mono text-lg font-bold ${isCyan ? 'text-[#00f0ff]' : 'text-[#ff0066]'}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-3 bg-[var(--bg-deep)] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out relative ${
            isCyan
              ? 'bg-gradient-to-r from-[#00a8b3] to-[#00f0ff] shadow-[0_0_20px_rgba(0,240,255,0.4)]'
              : 'bg-gradient-to-r from-[#b30047] to-[#ff0066] shadow-[0_0_20px_rgba(255,0,102,0.4)]'
          }`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

// Radial chart for confidence
function ConfidenceGauge({ value }: { value: number }) {
  const radius = 60;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: 140, height: 140 }}>
        <svg className="transform -rotate-90" width={140} height={140}>
          <circle
            cx={70}
            cy={70}
            r={radius}
            fill="none"
            stroke="var(--bg-deep)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={70}
            cy={70}
            r={radius}
            fill="none"
            stroke="var(--accent-purple)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{
              transition: 'stroke-dashoffset 1.5s cubic-bezier(0.16, 1, 0.3, 1)',
              filter: 'drop-shadow(0 0 10px var(--accent-purple))'
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl font-bold text-[var(--text-primary)]">
            {value.toFixed(1)}%
          </span>
          <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider mt-1">
            Confidence
          </span>
        </div>
      </div>
    </div>
  );
}

export default function PredictionForm() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [teams, setTeams] = useState<Team[]>([]);
  const [loadingTeams, setLoadingTeams] = useState(true);

  const [formData, setFormData] = useState<MatchInput>({
    home_team: '',
    away_team: '',
    model: 'random_forest',
  });

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        const apiUrl = import.meta.env.PUBLIC_API_URL || 'https://dspro1--epl-predictor-fastapi-app.modal.run';
        const response = await fetch(`${apiUrl}/teams`);
        if (!response.ok) {
          throw new Error('Failed to fetch teams');
        }
        const data = await response.json();
        setTeams(data.teams);
      } catch (err) {
        console.error('Error fetching teams:', err);
      } finally {
        setLoadingTeams(false);
      }
    };

    fetchTeams();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = import.meta.env.PUBLIC_API_URL || 'https://dspro1--epl-predictor-fastapi-app.modal.run';
      const apiKey = import.meta.env.PUBLIC_API_KEY;

      if (!apiKey) {
        throw new Error('API key not configured');
      }

      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof MatchInput, value: string | number | undefined) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const modelLabels: Record<string, string> = {
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
  };

  return (
    <div className="space-y-6">
      {/* Form Card */}
      <form onSubmit={handleSubmit} className="card-dark card-glow p-6 md:p-8">
        <div className="space-y-6">
          {/* Model Selection */}
          <div>
            <label className="label-dark">
              <span className="text-[var(--accent-cyan)]">&gt;</span> Select Model
            </label>
            <div className="grid grid-cols-3 gap-2">
              {['random_forest', 'xgboost', 'lightgbm'].map((model) => (
                <button
                  key={model}
                  type="button"
                  onClick={() => handleInputChange('model', model)}
                  className={`
                    py-3 px-4 rounded-lg font-mono text-sm transition-all border
                    ${formData.model === model
                      ? 'bg-[var(--accent-cyan)]/10 border-[var(--accent-cyan)]/50 text-[var(--accent-cyan)]'
                      : 'bg-[var(--bg-deep)] border-[var(--border-subtle)] text-[var(--text-secondary)] hover:border-[var(--text-muted)]'
                    }
                  `}
                >
                  {model === 'random_forest' ? 'RF' : model === 'xgboost' ? 'XGB' : 'LGBM'}
                </button>
              ))}
            </div>
          </div>

          {/* Team Selection */}
          {loadingTeams ? (
            <div className="flex items-center justify-center py-8 gap-3">
              <div className="spinner" />
              <span className="font-mono text-sm text-[var(--text-muted)]">
                Loading teams<span className="cursor-blink"></span>
              </span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label-dark">
                  <span className="text-[var(--accent-cyan)]">&gt;</span> Home Team
                </label>
                <select
                  required
                  value={formData.home_team}
                  onChange={(e) => handleInputChange('home_team', e.target.value)}
                  className="input-dark select-dark"
                >
                  <option value="">Select team...</option>
                  {teams.map(team => (
                    <option key={team.team_name} value={team.team_name}>
                      {team.team_name} ({Math.round(team.elo_rating)})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="label-dark">
                  <span className="text-[var(--accent-magenta)]">&gt;</span> Away Team
                </label>
                <select
                  required
                  value={formData.away_team}
                  onChange={(e) => handleInputChange('away_team', e.target.value)}
                  className="input-dark select-dark"
                >
                  <option value="">Select team...</option>
                  {teams.map(team => (
                    <option key={team.team_name} value={team.team_name}>
                      {team.team_name} ({Math.round(team.elo_rating)})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] transition-colors font-mono text-sm"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 p-4 bg-[var(--bg-deep)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-xs text-[var(--text-muted)]">
                // Override auto-fetched values for what-if scenarios
              </p>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <label className="label-dark">Home ELO</label>
                  <input
                    type="number"
                    value={formData.home_elo || ''}
                    onChange={(e) => handleInputChange('home_elo', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="10"
                  />
                </div>

                <div>
                  <label className="label-dark">Away ELO</label>
                  <input
                    type="number"
                    value={formData.away_elo || ''}
                    onChange={(e) => handleInputChange('away_elo', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="10"
                  />
                </div>

                <div>
                  <label className="label-dark">Home GF Avg</label>
                  <input
                    type="number"
                    value={formData.home_gf_roll || ''}
                    onChange={(e) => handleInputChange('home_gf_roll', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="label-dark">Away GF Avg</label>
                  <input
                    type="number"
                    value={formData.away_gf_roll || ''}
                    onChange={(e) => handleInputChange('away_gf_roll', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="label-dark">Home Rest Days</label>
                  <input
                    type="number"
                    value={formData.rest_days_home || ''}
                    onChange={(e) => handleInputChange('rest_days_home', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="1"
                  />
                </div>

                <div>
                  <label className="label-dark">Away Rest Days</label>
                  <input
                    type="number"
                    value={formData.rest_days_away || ''}
                    onChange={(e) => handleInputChange('rest_days_away', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="input-dark"
                    placeholder="Auto"
                    step="1"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !formData.home_team || !formData.away_team}
            className="btn-primary w-full"
          >
            {loading ? (
              <>
                <div className="spinner w-5 h-5" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Run Prediction</span>
              </>
            )}
          </button>

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-[var(--accent-magenta)]/10 border border-[var(--accent-magenta)]/30 rounded-lg">
              <p className="font-mono text-sm text-[var(--accent-magenta)]">
                <span className="opacity-60">Error:</span> {error}
              </p>
            </div>
          )}
        </div>
      </form>

      {/* Results Display */}
      {result && (
        <div className="card-dark card-glow-cyan overflow-hidden animate-scale-in">
          {/* Top gradient bar */}
          <div className="h-1 bg-gradient-to-r from-[var(--accent-cyan)] via-[var(--accent-purple)] to-[var(--accent-magenta)]" />

          <div className="p-6 md:p-8">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-[var(--accent-green)]">
                  <span className="opacity-60">&gt;</span> prediction_complete
                </span>
              </div>
              <span className="px-3 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded-full font-mono text-xs text-[var(--text-secondary)]">
                {modelLabels[result.model_used || formData.model || 'unknown']}
              </span>
            </div>

            {/* Match Title */}
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-4 md:gap-6 mb-4">
                <span className="text-xl md:text-2xl font-bold text-[var(--text-primary)]">
                  {result.home_team}
                </span>
                <span className="font-mono text-sm text-[var(--text-muted)]">vs</span>
                <span className="text-xl md:text-2xl font-bold text-[var(--text-primary)]">
                  {result.away_team}
                </span>
              </div>

              {/* Prediction Outcome */}
              <div className="inline-block px-6 py-3 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                  Predicted Outcome
                </p>
                <p className={`text-2xl md:text-3xl font-bold ${
                  result.prediction.includes('Away')
                    ? 'text-[var(--accent-magenta)]'
                    : 'text-[var(--accent-cyan)]'
                }`}>
                  {result.prediction}
                </p>
              </div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {/* Probability Bars */}
              <div className="md:col-span-2 space-y-4">
                <ProbabilityBar
                  label="Home/Draw"
                  value={result.probabilities.home_or_draw}
                  type="cyan"
                />
                <ProbabilityBar
                  label="Away Win"
                  value={result.probabilities.away}
                  type="magenta"
                />
              </div>

              {/* Confidence Gauge */}
              <div className="flex items-center justify-center">
                <ConfidenceGauge value={result.confidence * 100} />
              </div>
            </div>

            {/* Features Used */}
            <details className="group">
              <summary className="cursor-pointer font-mono text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors flex items-center gap-2">
                <svg
                  className="w-4 h-4 transition-transform group-open:rotate-90"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                View Features Used ({Object.keys(result.features_used).length})
              </summary>
              <div className="mt-4 p-4 bg-[var(--bg-deep)] rounded-xl border border-[var(--border-subtle)] overflow-x-auto">
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {Object.entries(result.features_used).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center gap-2">
                      <span className="font-mono text-xs text-[var(--text-muted)] truncate">
                        {key}
                      </span>
                      <span className="font-mono text-xs text-[var(--accent-cyan)]">
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
