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
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

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
      // Check cache first
      const cached = checkCache();
      if (cached) {
        setFixture(cached.fixture);
        setPredictions(cached.predictions);
        setLoading(false);
        return;
      }

      // Fetch next fixture
      const apiUrl = import.meta.env.PUBLIC_API_URL || 'https://dspro1--epl-predictor-fastapi-app.modal.run';

      const fixtureResponse = await fetch(`${apiUrl}/fixtures/next`);
      if (!fixtureResponse.ok) {
        if (fixtureResponse.status === 404) {
          // Silently fail - no upcoming fixtures available
          // Component will not render anything
          setLoading(false);
          return;
        }
        throw new Error('Failed to fetch next fixture');
      }

      const fixtureData: Fixture = await fixtureResponse.json();
      setFixture(fixtureData);

      // Make predictions from all three models
      const apiKey = import.meta.env.PUBLIC_API_KEY;
      if (!apiKey) {
        throw new Error('API key not configured');
      }

      const models = ['random_forest', 'xgboost', 'lightgbm'];
      const predictionsData: Record<string, PredictionResult> = {};

      for (const model of models) {
        const predictionResponse = await fetch(`${apiUrl}/predict`, {
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

        if (predictionResponse.ok) {
          const predictionData: PredictionResult = await predictionResponse.json();
          predictionsData[model] = predictionData;
        }
      }

      setPredictions(predictionsData);

      // Cache the results
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
      const cached = sessionStorage.getItem(CACHE_KEY);
      if (!cached) return null;

      const data: CachedData = JSON.parse(cached);
      const now = Date.now();

      // Check if cache is still valid
      if (now - data.timestamp < CACHE_DURATION) {
        return data;
      }

      // Cache expired, remove it
      sessionStorage.removeItem(CACHE_KEY);
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
      sessionStorage.setItem(CACHE_KEY, JSON.stringify(data));
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
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div>
          <p className="text-gray-600">Loading next match prediction...</p>
        </div>
      </div>
    );
  }

  if (error || !fixture || Object.keys(predictions).length === 0) {
    return null; // Don't show anything if there's an error or no data
  }

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl shadow-xl p-8 mb-8 border-2 border-indigo-200">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <div className="bg-indigo-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
            NEXT MATCH
          </div>
          <div className="text-sm text-gray-600">
            Auto-predicted
          </div>
        </div>
        <div className="text-sm text-gray-600">
          MW {fixture.matchweek}
        </div>
      </div>

      <div className="text-center mb-6">
        <h3 className="text-3xl font-bold text-gray-900 mb-2">
          {fixture.home_team} vs {fixture.away_team}
        </h3>
        <p className="text-sm text-gray-600 mb-1">
          {formatKickoffTime(fixture.kickoff)}
        </p>
        {fixture.venue && (
          <p className="text-xs text-gray-500">{fixture.venue}</p>
        )}
      </div>

      {Object.entries(predictions).map(([model, pred]) => (
        <div key={model} className="bg-white rounded-xl p-6 mb-4">
          <div className="flex items-center justify-between mb-4">
            <div className="text-center flex-1">
              <p className="text-2xl font-bold text-indigo-600 mb-2">
                {pred.prediction}
              </p>
              <p className="text-lg text-gray-600">
                Confidence: <span className="font-bold text-purple-600">
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>
            <span className="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm font-medium">
              {pred.model_used.replace('_', ' ').toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
              <p className="text-xs text-gray-600 mb-1">Home/Draw</p>
              <p className="text-2xl font-bold text-green-700">
                {(pred.probabilities.home_or_draw * 100).toFixed(1)}%
              </p>
              <div className="mt-2 bg-green-200 rounded-full h-1.5">
                <div
                  className="bg-green-600 h-1.5 rounded-full transition-all duration-500"
                  style={{ width: `${pred.probabilities.home_or_draw * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg">
              <p className="text-xs text-gray-600 mb-1">Away Win</p>
              <p className="text-2xl font-bold text-orange-700">
                {(pred.probabilities.away * 100).toFixed(1)}%
              </p>
              <div className="mt-2 bg-orange-200 rounded-full h-1.5">
                <div
                  className="bg-orange-600 h-1.5 rounded-full transition-all duration-500"
                  style={{ width: `${pred.probabilities.away * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      ))}

      <p className="text-xs text-center text-gray-500">
        Prediction updates automatically every 5 minutes
      </p>
    </div>
  );
}
