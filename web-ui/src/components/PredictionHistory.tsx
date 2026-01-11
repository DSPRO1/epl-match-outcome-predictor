import { useState, useEffect, useRef, useCallback } from 'react';

interface PredictionRecord {
  id: number;
  home_team: string;
  away_team: string;
  prediction: string;
  home_or_draw_prob: number | null;
  away_prob: number | null;
  confidence: number | null;
  model_used: string | null;
  match_date: string | null;
  created_at: string | null;
  actual_home_score: number | null;
  actual_away_score: number | null;
  actual_result: string | null;
  was_correct: boolean | null;
}

interface HistoryResponse {
  count: number;
  total_count: number;
  offset: number;
  has_more: boolean;
  total_with_results: number;
  correct_predictions: number;
  accuracy: number | null;
  predictions: PredictionRecord[];
  message?: string;
}

const MODELS = [
  { value: '', label: 'All Models' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'lightgbm', label: 'LightGBM' },
];

const PAGE_SIZE = 20;
const CACHE_KEY_PREFIX = 'epl_prediction_history';
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

interface CachedHistoryData {
  predictions: PredictionRecord[];
  stats: {
    total_count: number;
    total_with_results: number;
    correct_predictions: number;
    accuracy: number | null;
    has_more: boolean;
  };
  timestamp: number;
}

const getCacheKey = (model: string) => `${CACHE_KEY_PREFIX}_${model || 'all'}`;

const getCache = (model: string): CachedHistoryData | null => {
  try {
    const cached = localStorage.getItem(getCacheKey(model));
    if (!cached) return null;

    const data: CachedHistoryData = JSON.parse(cached);
    if (Date.now() - data.timestamp < CACHE_DURATION) {
      return data;
    }
    localStorage.removeItem(getCacheKey(model));
    return null;
  } catch {
    return null;
  }
};

const setCache = (model: string, predictions: PredictionRecord[], stats: CachedHistoryData['stats']) => {
  try {
    const data: CachedHistoryData = { predictions, stats, timestamp: Date.now() };
    localStorage.setItem(getCacheKey(model), JSON.stringify(data));
  } catch {
    // Ignore cache errors
  }
};

// Fetch history for a specific model filter
const fetchModelHistory = async (model: string, offset = 0): Promise<HistoryResponse | null> => {
  try {
    const apiUrl = import.meta.env.PUBLIC_API_URL || 'https://dspro1--epl-predictor-fastapi-app.modal.run';
    const params = new URLSearchParams({
      limit: PAGE_SIZE.toString(),
      offset: offset.toString(),
    });

    if (model) {
      params.append('model', model);
    }

    const response = await fetch(`${apiUrl}/predictions/history?${params}`);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
};

export default function PredictionHistory() {
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
  const [stats, setStats] = useState<{
    total_count: number;
    total_with_results: number;
    correct_predictions: number;
    accuracy: number | null;
    has_more: boolean;
  } | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [offset, setOffset] = useState(0);
  const [preloaded, setPreloaded] = useState(false);

  const observerRef = useRef<IntersectionObserver | null>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Preload all model filters on mount
  useEffect(() => {
    const preloadAllModels = async () => {
      const modelFilters = ['', 'random_forest', 'xgboost', 'lightgbm'];

      // Check if all are already cached
      const allCached = modelFilters.every(m => getCache(m) !== null);
      if (allCached) {
        // Use cached data for initial view
        const cached = getCache('');
        if (cached) {
          setPredictions(cached.predictions);
          setStats(cached.stats);
          setOffset(cached.predictions.length);
        }
        setLoading(false);
        setPreloaded(true);
        return;
      }

      // Fetch all models in parallel
      const results = await Promise.all(
        modelFilters.map(async (model) => {
          // Skip if already cached
          if (getCache(model)) return { model, data: null, fromCache: true };
          const data = await fetchModelHistory(model);
          return { model, data, fromCache: false };
        })
      );

      // Cache the results
      for (const { model, data, fromCache } of results) {
        if (!fromCache && data) {
          const newStats = {
            total_count: data.total_count,
            total_with_results: data.total_with_results,
            correct_predictions: data.correct_predictions,
            accuracy: data.accuracy,
            has_more: data.has_more,
          };
          setCache(model, data.predictions, newStats);
        }
      }

      // Set initial view (all models)
      const cached = getCache('');
      if (cached) {
        setPredictions(cached.predictions);
        setStats(cached.stats);
        setOffset(cached.predictions.length);
      }

      setLoading(false);
      setPreloaded(true);
    };

    preloadAllModels();
  }, []);

  // Handle model filter change - use cached data
  useEffect(() => {
    if (!preloaded) return;

    const cached = getCache(selectedModel);
    if (cached) {
      setPredictions(cached.predictions);
      setStats(cached.stats);
      setOffset(cached.predictions.length);
    }
  }, [selectedModel, preloaded]);

  // Load more for infinite scroll
  const loadMore = useCallback(async () => {
    if (loadingMore) return;

    setLoadingMore(true);
    try {
      const data = await fetchModelHistory(selectedModel, offset);
      if (data) {
        const allPredictions = [...predictions, ...data.predictions];
        const newStats = {
          total_count: data.total_count,
          total_with_results: data.total_with_results,
          correct_predictions: data.correct_predictions,
          accuracy: data.accuracy,
          has_more: data.has_more,
        };
        setPredictions(allPredictions);
        setStats(newStats);
        setOffset(offset + data.predictions.length);
        // Update cache with all loaded predictions
        setCache(selectedModel, allPredictions, newStats);
      }
    } catch (err) {
      console.error('Error loading more:', err);
    } finally {
      setLoadingMore(false);
    }
  }, [offset, selectedModel, predictions, loadingMore]);

  // Infinite scroll observer
  useEffect(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && stats?.has_more && !loadingMore && !loading) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observerRef.current.observe(loadMoreRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [stats?.has_more, loadingMore, loading, loadMore]);

  const formatMatchDate = (dateStr: string | null) => {
    if (!dateStr) return '-';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-GB', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  const getModelLabel = (model: string | null) => {
    const labels: Record<string, string> = {
      'random_forest': 'RF',
      'xgboost': 'XGB',
      'lightgbm': 'LGBM',
    };
    return model ? labels[model] || model : '-';
  };

  const getModelColor = (model: string | null) => {
    const colors: Record<string, string> = {
      'random_forest': 'var(--accent-cyan)',
      'xgboost': 'var(--accent-purple)',
      'lightgbm': 'var(--accent-green)',
    };
    return model ? colors[model] || 'var(--text-secondary)' : 'var(--text-secondary)';
  };

  if (loading && predictions.length === 0) {
    return (
      <div className="card-dark p-8">
        <div className="flex items-center justify-center gap-3">
          <div className="spinner spinner-glow" />
          <span className="text-[var(--text-secondary)] font-mono text-sm">
            Loading prediction history<span className="cursor-blink"></span>
          </span>
        </div>
      </div>
    );
  }

  if (error && predictions.length === 0) {
    return (
      <div className="card-dark p-8">
        <div className="text-center">
          <p className="text-[var(--text-muted)] font-mono text-sm">{error}</p>
          <button
            onClick={() => loadHistory(true)}
            className="mt-4 px-4 py-2 text-sm font-mono text-[var(--accent-cyan)] hover:text-[var(--text-primary)] transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="card-dark overflow-hidden">
      {/* Top gradient bar */}
      <div className="h-1 bg-gradient-to-r from-[var(--accent-purple)] via-[var(--accent-cyan)] to-[var(--accent-green)]" />

      <div className="p-6">
        {/* Header with stats and filter */}
        <div className="flex flex-col gap-4 mb-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--accent-purple)]/10 border border-[var(--accent-purple)]/30 rounded-full">
                <svg className="w-4 h-4 text-[var(--accent-purple)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="font-mono text-xs text-[var(--accent-purple)] uppercase tracking-wider">
                  Track Record
                </span>
              </div>
            </div>

            {/* Model filter */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-[var(--text-muted)]">Filter:</span>
              <div className="flex gap-1">
                {MODELS.map((model) => (
                  <button
                    key={model.value}
                    onClick={() => setSelectedModel(model.value)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                      selectedModel === model.value
                        ? 'bg-[var(--accent-cyan)]/20 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/50'
                        : 'bg-[var(--bg-elevated)] text-[var(--text-muted)] border border-[var(--border-subtle)] hover:border-[var(--border-default)]'
                    }`}
                  >
                    {model.value ? getModelLabel(model.value) : 'All'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Accuracy stats */}
          {stats && stats.total_with_results > 0 && (
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-elevated)] rounded-lg border border-[var(--border-subtle)]">
                <span className="text-xs text-[var(--text-muted)]">Total:</span>
                <span className="font-mono text-sm text-[var(--text-primary)]">
                  {stats.total_count}
                </span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-elevated)] rounded-lg border border-[var(--border-subtle)]">
                <span className="text-xs text-[var(--text-muted)]">Correct:</span>
                <span className="font-mono text-sm text-[var(--text-primary)]">
                  {stats.correct_predictions}/{stats.total_with_results}
                </span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--accent-green)]/10 rounded-lg border border-[var(--accent-green)]/30">
                <span className="text-xs text-[var(--text-muted)]">Accuracy:</span>
                <span className="font-mono text-sm font-bold text-[var(--accent-green)]">
                  {stats.accuracy?.toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Predictions list */}
        {predictions.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)] flex items-center justify-center">
              <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
              </svg>
            </div>
            <p className="text-[var(--text-secondary)] font-medium mb-2">No Predictions Found</p>
            <p className="text-[var(--text-muted)] text-sm">
              {selectedModel ? `No predictions for ${selectedModel}` : 'No predictions recorded yet'}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {predictions.map((pred, idx) => (
              <div
                key={`${pred.id}-${idx}`}
                className="group p-4 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] hover:border-[var(--border-default)] transition-all"
              >
                <div className="flex flex-col md:flex-row md:items-center gap-3">
                  {/* Match info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-[var(--text-primary)] truncate">
                        {pred.home_team}
                      </span>
                      <span className="text-xs text-[var(--text-muted)]">vs</span>
                      <span className="text-sm font-medium text-[var(--text-primary)] truncate">
                        {pred.away_team}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
                      <span className="font-mono">{formatMatchDate(pred.match_date)}</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium"
                        style={{
                          backgroundColor: `color-mix(in srgb, ${getModelColor(pred.model_used)} 15%, transparent)`,
                          color: getModelColor(pred.model_used),
                        }}
                      >
                        {getModelLabel(pred.model_used)}
                      </span>
                    </div>
                  </div>

                  {/* Prediction */}
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <span className={`text-sm font-bold ${
                        pred.prediction === 'Away Win'
                          ? 'text-[var(--accent-magenta)]'
                          : 'text-[var(--accent-cyan)]'
                      }`}>
                        {pred.prediction === 'Home Win or Draw' ? 'H/D' : 'Away'}
                      </span>
                      <div className="text-[10px] text-[var(--text-muted)] font-mono">
                        {pred.confidence ? `${(pred.confidence * 100).toFixed(0)}%` : '-'}
                      </div>
                    </div>

                    {/* Actual result */}
                    <div className="text-center min-w-[60px]">
                      {pred.actual_home_score !== null && pred.actual_away_score !== null ? (
                        <>
                          <span className="font-mono text-sm font-bold text-[var(--text-primary)]">
                            {pred.actual_home_score}-{pred.actual_away_score}
                          </span>
                          <div className="text-[10px] text-[var(--text-muted)]">
                            {pred.actual_result}
                          </div>
                        </>
                      ) : (
                        <span className="text-xs text-[var(--text-muted)]">Pending</span>
                      )}
                    </div>

                    {/* Result indicator */}
                    <div className="w-8 h-8 flex items-center justify-center">
                      {pred.was_correct === null ? (
                        <div className="w-6 h-6 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)] flex items-center justify-center">
                          <span className="text-[8px] font-mono text-[var(--text-muted)]">?</span>
                        </div>
                      ) : pred.was_correct ? (
                        <div className="w-6 h-6 rounded-full bg-[var(--accent-green)]/20 border border-[var(--accent-green)]/50 flex items-center justify-center">
                          <svg className="w-3.5 h-3.5 text-[var(--accent-green)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-[var(--accent-magenta)]/20 border border-[var(--accent-magenta)]/50 flex items-center justify-center">
                          <svg className="w-3.5 h-3.5 text-[var(--accent-magenta)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Load more trigger */}
            <div ref={loadMoreRef} className="h-4" />

            {/* Loading more indicator */}
            {loadingMore && (
              <div className="flex items-center justify-center py-4 gap-2">
                <div className="spinner" />
                <span className="text-xs text-[var(--text-muted)] font-mono">Loading more...</span>
              </div>
            )}

            {/* End of list */}
            {!stats?.has_more && predictions.length > 0 && (
              <div className="text-center py-4">
                <span className="text-xs text-[var(--text-muted)] font-mono">
                  Showing all {predictions.length} predictions
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
