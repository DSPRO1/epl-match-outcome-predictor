import { useState, useEffect } from 'react';
import IntroStory from './IntroStory';
import NextMatchPrediction from './NextMatchPrediction';
import PredictionForm from './PredictionForm';
import PredictionHistory from './PredictionHistory';

const INTRO_SEEN_KEY = 'epl_intro_seen';

export default function AppWrapper() {
  const [showIntro, setShowIntro] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user has seen intro before
    const hasSeenIntro = sessionStorage.getItem(INTRO_SEEN_KEY);
    if (hasSeenIntro) {
      setShowIntro(false);
    }
    setIsLoading(false);
  }, []);

  const handleIntroComplete = () => {
    sessionStorage.setItem(INTRO_SEEN_KEY, 'true');
    setShowIntro(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="spinner spinner-glow" />
      </div>
    );
  }

  if (showIntro) {
    return <IntroStory onComplete={handleIntroComplete} />;
  }

  return (
    <main className="min-h-screen px-4 py-8 md:py-12">
      {/* Header Section */}
      <header className="max-w-5xl mx-auto mb-12 md:mb-16 animate-fade-in-up">
        {/* Terminal-style header badge */}
        <div className="flex items-center justify-center gap-2 mb-6">
          <span className="font-mono text-xs text-[var(--text-muted)] tracking-wider">
            <span className="text-[var(--accent-cyan)]">&gt;</span> epl_predictor v2.1.0
          </span>
          <span className="w-2 h-2 rounded-full bg-[var(--accent-green)] animate-pulse"></span>
          <span className="font-mono text-xs text-[var(--accent-green)]">LIVE</span>
        </div>

        {/* Main Title */}
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4 tracking-tight">
            <span className="text-gradient-cyan">EPL</span>
            <span className="text-[var(--text-primary)]"> Match</span>
            <br className="md:hidden" />
            <span className="text-[var(--text-secondary)]"> Predictor</span>
          </h1>
          <p className="text-[var(--text-secondary)] text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Machine learning predictions powered by
            <span className="text-[var(--accent-cyan)] font-mono"> 11 seasons</span> of Premier League data
          </p>
        </div>

        {/* Stats Row */}
        <div className="flex flex-wrap justify-center gap-4 mt-8">
          <div className="stat-badge opacity-0 animate-fade-in-up delay-100" style={{ animationFillMode: 'forwards' }}>
            <span className="stat-value">71.05%</span>
            <span className="stat-label">Accuracy</span>
          </div>
          <div className="stat-badge opacity-0 animate-fade-in-up delay-200" style={{ animationFillMode: 'forwards' }}>
            <span className="stat-value" style={{ color: 'var(--accent-purple)' }}>3</span>
            <span className="stat-label">ML Models</span>
          </div>
          <div className="stat-badge opacity-0 animate-fade-in-up delay-300" style={{ animationFillMode: 'forwards' }}>
            <span className="stat-value" style={{ color: 'var(--accent-green)' }}>14</span>
            <span className="stat-label">Features</span>
          </div>
          <div className="stat-badge opacity-0 animate-fade-in-up delay-400" style={{ animationFillMode: 'forwards' }}>
            <span className="stat-value" style={{ color: 'var(--accent-orange)' }}>4,180+</span>
            <span className="stat-label">Matches</span>
          </div>
        </div>

        {/* Model Pills */}
        <div className="flex flex-wrap justify-center gap-2 mt-6">
          <span className="model-chip">
            <span className="model-chip-dot"></span>
            Random Forest
          </span>
          <span className="model-chip">
            <span className="model-chip-dot"></span>
            XGBoost
          </span>
          <span className="model-chip">
            <span className="model-chip-dot"></span>
            LightGBM
          </span>
        </div>

        {/* View methodology link */}
        <div className="flex justify-center mt-6">
          <button
            onClick={() => {
              sessionStorage.removeItem(INTRO_SEEN_KEY);
              setShowIntro(true);
            }}
            className="text-sm font-mono text-[var(--text-muted)] hover:text-[var(--accent-cyan)] transition-colors"
          >
            View methodology →
          </button>
        </div>
      </header>

      {/* Next Match Prediction Section */}
      <section className="max-w-4xl mx-auto mb-12 opacity-0 animate-fade-in-up delay-300" style={{ animationFillMode: 'forwards' }}>
        <NextMatchPrediction />
      </section>

      {/* Divider */}
      <div className="max-w-4xl mx-auto mb-12">
        <div className="flex items-center gap-4">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent"></div>
          <span className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-widest">Custom Analysis</span>
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent"></div>
        </div>
      </div>

      {/* Prediction Form Section */}
      <section className="max-w-4xl mx-auto mb-16 opacity-0 animate-fade-in-up delay-400" style={{ animationFillMode: 'forwards' }}>
        <div className="text-center mb-8">
          <h2 className="text-2xl md:text-3xl font-bold text-[var(--text-primary)] mb-2">
            Run Your Own Prediction
          </h2>
          <p className="text-[var(--text-secondary)]">
            Select any two teams and choose your model
          </p>
        </div>
        <PredictionForm />
      </section>

      {/* Divider */}
      <div className="max-w-4xl mx-auto mb-12">
        <div className="flex items-center gap-4">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent"></div>
          <span className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-widest">Track Record</span>
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent"></div>
        </div>
      </div>

      {/* Prediction History Section */}
      <section className="max-w-5xl mx-auto mb-16 opacity-0 animate-fade-in-up delay-500" style={{ animationFillMode: 'forwards' }}>
        <div className="text-center mb-8">
          <h2 className="text-2xl md:text-3xl font-bold text-[var(--text-primary)] mb-2">
            Predictions vs Reality
          </h2>
          <p className="text-[var(--text-secondary)]">
            See how our models performed against actual match results
          </p>
        </div>
        <PredictionHistory />
      </section>

      {/* Footer */}
      <footer className="max-w-4xl mx-auto text-center pb-8">
        <div className="card-dark p-6 inline-block">
          <div className="flex items-center justify-center gap-6 text-sm">
            <span className="font-mono text-[var(--text-muted)]">
              <span className="text-[var(--accent-cyan)]">&gt;</span> EPL Data 2014-2025
            </span>
            <span className="w-1 h-1 rounded-full bg-[var(--text-muted)]"></span>
            <a
              href="https://github.com/DSPRO1/epl-match-outcome-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View Source
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
