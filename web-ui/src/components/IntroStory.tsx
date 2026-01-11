import { useState } from 'react';

interface StorySection {
  id: string;
  label: string;
  title: string;
  content: React.ReactNode;
}

export default function IntroStory({ onComplete }: { onComplete: () => void }) {
  const [currentSection, setCurrentSection] = useState(0);

  const sections: StorySection[] = [
    {
      id: 'problem',
      label: '01',
      title: 'The Challenge',
      content: (
        <div className="space-y-6">
          <p className="text-lg text-[var(--text-secondary)] leading-relaxed">
            Predicting football match outcomes is one of the hardest problems in sports analytics.
            High randomness, dynamic team performance, and the infamous <span className="text-[var(--accent-cyan)]">"draw problem"</span> make
            accurate predictions extremely difficult.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-magenta)]">54.6%</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Initial 3-class accuracy</p>
            </div>
            <div className="p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-orange)]">~25%</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Matches end in draws</p>
            </div>
            <div className="p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--text-muted)]">3/221</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Draws correctly predicted</p>
            </div>
          </div>
          <p className="text-sm text-[var(--text-muted)] italic">
            Our initial model could barely predict draws at all — essentially random guessing.
          </p>
        </div>
      ),
    },
    {
      id: 'solution',
      label: '02',
      title: 'The Breakthrough',
      content: (
        <div className="space-y-6">
          <p className="text-lg text-[var(--text-secondary)] leading-relaxed">
            We reformulated the problem. Instead of predicting three outcomes (Home/Draw/Away),
            we focused on a binary question: <span className="text-[var(--accent-cyan)]">"Will the away team win?"</span>
          </p>
          <div className="flex items-center justify-center gap-4 my-8">
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--accent-magenta)]/30 opacity-50">
              <p className="font-mono text-sm text-[var(--text-muted)]">3-CLASS</p>
              <p className="text-lg text-[var(--text-secondary)] mt-2">H / D / A</p>
            </div>
            <svg className="w-8 h-8 text-[var(--accent-cyan)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--accent-cyan)]/30">
              <p className="font-mono text-sm text-[var(--accent-cyan)]">BINARY</p>
              <p className="text-lg text-[var(--text-primary)] mt-2">Home/Draw vs Away</p>
            </div>
          </div>
          <div className="p-4 bg-[var(--accent-cyan)]/5 rounded-xl border border-[var(--accent-cyan)]/20">
            <p className="text-sm text-[var(--text-secondary)]">
              <span className="text-[var(--accent-cyan)] font-semibold">Result:</span> Accuracy jumped from 54.6% to over 70% with the same features.
            </p>
          </div>
        </div>
      ),
    },
    {
      id: 'features',
      label: '03',
      title: 'The Features',
      content: (
        <div className="space-y-6">
          <p className="text-lg text-[var(--text-secondary)] leading-relaxed">
            We engineered <span className="text-[var(--accent-cyan)]">14 predictive features</span> from raw match data,
            capturing team strength, form, and historical matchups.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
            {[
              { name: 'ELO Rating', desc: 'Dynamic team strength', color: 'cyan' },
              { name: 'ELO Diff', desc: 'Relative advantage', color: 'cyan' },
              { name: 'Goals For', desc: '5-match rolling avg', color: 'green' },
              { name: 'Goals Against', desc: '5-match rolling avg', color: 'green' },
              { name: 'Points Avg', desc: 'Recent form', color: 'green' },
              { name: 'Rest Days', desc: 'Recovery time', color: 'purple' },
              { name: 'H2H Home', desc: 'Head-to-head history', color: 'orange' },
              { name: 'H2H Away', desc: 'Head-to-head history', color: 'orange' },
            ].map((feature) => (
              <div
                key={feature.name}
                className="p-3 bg-[var(--bg-elevated)] rounded-lg border border-[var(--border-subtle)]"
              >
                <p className={`font-mono text-xs text-[var(--accent-${feature.color})]`}>{feature.name}</p>
                <p className="text-xs text-[var(--text-muted)] mt-1">{feature.desc}</p>
              </div>
            ))}
          </div>
          <p className="text-sm text-[var(--text-muted)]">
            Features are calculated for both home and away teams, updated after every match.
          </p>
        </div>
      ),
    },
    {
      id: 'models',
      label: '04',
      title: 'The Models',
      content: (
        <div className="space-y-6">
          <p className="text-lg text-[var(--text-secondary)] leading-relaxed">
            We trained and compared three ensemble models using time-series cross-validation
            across <span className="text-[var(--accent-cyan)]">11 EPL seasons</span>.
          </p>
          <div className="space-y-4 mt-6">
            {[
              { name: 'Random Forest', accuracy: 69.95, logloss: 0.5865, precision: 64.98, recall: 60.45, selected: false },
              { name: 'XGBoost', accuracy: 71.26, logloss: 0.5663, precision: 61.52, recall: 32.32, selected: false },
              { name: 'LightGBM', accuracy: 71.05, logloss: 0.5850, precision: 66.49, recall: 63.40, selected: true },
            ].map((model) => (
              <div
                key={model.name}
                className={`p-4 rounded-xl border ${
                  model.selected
                    ? 'bg-[var(--accent-cyan)]/5 border-[var(--accent-cyan)]/30'
                    : 'bg-[var(--bg-elevated)] border-[var(--border-subtle)]'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className={`font-mono text-sm ${model.selected ? 'text-[var(--accent-cyan)]' : 'text-[var(--text-secondary)]'}`}>
                    {model.name}
                  </span>
                  {model.selected && (
                    <span className="px-2 py-0.5 bg-[var(--accent-cyan)]/20 text-[var(--accent-cyan)] text-xs font-mono rounded">
                      SELECTED
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-4 gap-2 text-center">
                  <div>
                    <p className="font-mono text-lg text-[var(--text-primary)]">{model.accuracy}%</p>
                    <p className="text-xs text-[var(--text-muted)]">Accuracy</p>
                  </div>
                  <div>
                    <p className="font-mono text-lg text-[var(--text-primary)]">{model.logloss}</p>
                    <p className="text-xs text-[var(--text-muted)]">Log Loss</p>
                  </div>
                  <div>
                    <p className="font-mono text-lg text-[var(--text-primary)]">{model.precision}%</p>
                    <p className="text-xs text-[var(--text-muted)]">Precision</p>
                  </div>
                  <div>
                    <p className={`font-mono text-lg ${model.recall < 40 ? 'text-[var(--accent-magenta)]' : 'text-[var(--text-primary)]'}`}>
                      {model.recall}%
                    </p>
                    <p className="text-xs text-[var(--text-muted)]">Recall</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="text-sm text-[var(--text-muted)]">
            LightGBM offers the best balance — XGBoost has better log loss but misses 68% of away wins.
          </p>
        </div>
      ),
    },
    {
      id: 'result',
      label: '05',
      title: 'The Result',
      content: (
        <div className="space-y-6">
          <p className="text-lg text-[var(--text-secondary)] leading-relaxed">
            A production-grade prediction system trained on <span className="text-[var(--accent-cyan)]">4,000+ matches</span>,
            deployed serverlessly, and ready to predict in real-time.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-cyan)]">71%</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Accuracy</p>
            </div>
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-green)]">11</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Seasons</p>
            </div>
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-purple)]">14</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Features</p>
            </div>
            <div className="text-center p-4 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border-subtle)]">
              <p className="font-mono text-3xl font-bold text-[var(--accent-orange)]">3</p>
              <p className="text-sm text-[var(--text-muted)] mt-1">Models</p>
            </div>
          </div>
          <button
            onClick={onComplete}
            className="btn-primary w-full mt-8"
          >
            <span>Start Predicting</span>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        </div>
      ),
    },
  ];

  const currentSectionData = sections[currentSection];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Progress bar */}
      <div className="fixed top-0 left-0 right-0 h-1 bg-[var(--bg-elevated)] z-50">
        <div
          className="h-full bg-gradient-to-r from-[var(--accent-cyan)] to-[var(--accent-purple)] transition-all duration-500"
          style={{ width: `${((currentSection + 1) / sections.length) * 100}%` }}
        />
      </div>

      {/* Skip button */}
      <button
        onClick={onComplete}
        className="fixed top-6 right-6 z-50 px-4 py-2 text-sm font-mono text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
      >
        Skip intro →
      </button>

      {/* Main content */}
      <div className="flex-1 flex items-center justify-center p-6 md:p-12">
        <div className="max-w-2xl w-full">
          {/* Section label */}
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-sm text-[var(--accent-cyan)]">{currentSectionData.label}</span>
            <div className="flex-1 h-px bg-[var(--border-subtle)]" />
            <span className="font-mono text-xs text-[var(--text-muted)]">
              {currentSection + 1} / {sections.length}
            </span>
          </div>

          {/* Title */}
          <h2 className="text-3xl md:text-4xl font-bold text-[var(--text-primary)] mb-8">
            {currentSectionData.title}
          </h2>

          {/* Content */}
          <div className="mb-12">
            {currentSectionData.content}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentSection(Math.max(0, currentSection - 1))}
              disabled={currentSection === 0}
              className={`flex items-center gap-2 px-4 py-2 font-mono text-sm transition-colors ${
                currentSection === 0
                  ? 'text-[var(--text-muted)] cursor-not-allowed'
                  : 'text-[var(--text-secondary)] hover:text-[var(--accent-cyan)]'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back
            </button>

            {/* Section dots */}
            <div className="flex gap-2">
              {sections.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentSection(idx)}
                  className={`w-2 h-2 rounded-full transition-all ${
                    idx === currentSection
                      ? 'bg-[var(--accent-cyan)] w-6'
                      : idx < currentSection
                      ? 'bg-[var(--accent-cyan)]/50'
                      : 'bg-[var(--border-subtle)]'
                  }`}
                />
              ))}
            </div>

            {currentSection < sections.length - 1 ? (
              <button
                onClick={() => setCurrentSection(currentSection + 1)}
                className="flex items-center gap-2 px-4 py-2 font-mono text-sm text-[var(--accent-cyan)] hover:text-[var(--text-primary)] transition-colors"
              >
                Next
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            ) : (
              <div className="w-20" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
