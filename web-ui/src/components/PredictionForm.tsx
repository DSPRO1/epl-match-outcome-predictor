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

  // Fetch teams on component mount
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
        throw new Error('API key not configured. Please set PUBLIC_API_KEY in .env');
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

  const handleInputChange = (field: keyof MatchInput, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-800">Match Details</h2>

          {loadingTeams && (
            <div className="text-center py-4 text-gray-600">
              Loading teams...
            </div>
          )}

          {/* Team Selection */}
          {!loadingTeams && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Selection
                </label>
                <select
                  value={formData.model}
                  onChange={(e) => handleInputChange('model', e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                >
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="lightgbm">LightGBM</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Home Team
                  </label>
                  <select
                    required
                    value={formData.home_team}
                    onChange={(e) => handleInputChange('home_team', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                  >
                    <option value="">Select home team...</option>
                    {teams.map(team => (
                      <option key={team.team_name} value={team.team_name}>
                        {team.team_name} (ELO: {Math.round(team.elo_rating)})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Away Team
                  </label>
                  <select
                    required
                    value={formData.away_team}
                    onChange={(e) => handleInputChange('away_team', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                  >
                    <option value="">Select away team...</option>
                    {teams.map(team => (
                      <option key={team.team_name} value={team.team_name}>
                        {team.team_name} (ELO: {Math.round(team.elo_rating)})
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </>
          )}

          {/* Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            {showAdvanced ? '− Hide' : '+ Show'} Advanced Options
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-gray-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-700">Advanced Options</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Override team stats with custom values for "what-if" scenarios. Leave blank to use real-time stats from our database.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Home ELO Rating <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.home_elo || ''}
                    onChange={(e) => handleInputChange('home_elo', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-fetched from database"
                    step="10"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Away ELO Rating <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.away_elo || ''}
                    onChange={(e) => handleInputChange('away_elo', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-fetched from database"
                    step="10"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Home Goals For <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.home_gf_roll || ''}
                    onChange={(e) => handleInputChange('home_gf_roll', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-fetched from database"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Away Goals For <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.away_gf_roll || ''}
                    onChange={(e) => handleInputChange('away_gf_roll', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-fetched from database"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Home Rest Days <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.rest_days_home || ''}
                    onChange={(e) => handleInputChange('rest_days_home', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-calculated"
                    step="1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Away Rest Days <span className="text-gray-500 text-xs">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.rest_days_away || ''}
                    onChange={(e) => handleInputChange('rest_days_away', e.target.value ? parseFloat(e.target.value) : undefined)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Auto-calculated"
                    step="1"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Predicting...' : 'Predict Match Outcome'}
        </button>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}
      </form>

      {/* Results Display */}
      {result && (
        <div className="mt-6 bg-white rounded-2xl shadow-xl p-8 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-800">Prediction Results</h2>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
              {result.model_used.replace('_', ' ').toUpperCase()}
            </span>
          </div>

          <div className="text-center py-6">
            <h3 className="text-4xl font-bold text-gray-900 mb-2">
              {result.home_team} vs {result.away_team}
            </h3>
            <p className="text-2xl font-semibold text-blue-600 mb-4">
              {result.prediction}
            </p>
            <p className="text-lg text-gray-600">
              Confidence: <span className="font-bold text-purple-600">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Home/Draw Probability</p>
              <p className="text-3xl font-bold text-green-700">
                {(result.probabilities.home_or_draw * 100).toFixed(1)}%
              </p>
              <div className="mt-2 bg-green-200 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full"
                  style={{ width: `${result.probabilities.home_or_draw * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Away Win Probability</p>
              <p className="text-3xl font-bold text-orange-700">
                {(result.probabilities.away * 100).toFixed(1)}%
              </p>
              <div className="mt-2 bg-orange-200 rounded-full h-2">
                <div
                  className="bg-orange-600 h-2 rounded-full"
                  style={{ width: `${result.probabilities.away * 100}%` }}
                />
              </div>
            </div>
          </div>

          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">
              View Features Used
            </summary>
            <div className="mt-2 p-4 bg-gray-50 rounded-lg text-sm">
              <pre className="overflow-auto">
                {JSON.stringify(result.features_used, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
