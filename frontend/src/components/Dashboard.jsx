import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { getEvaluation, withRetry } from '../api'
import { MODEL_LABEL, humanizeFeature } from '../constants'

const MODELS = ['Logistic Regression', 'Random Forest', 'XGBoost', 'AdaBoost', 'KNN']

function mean(nums) {
  return nums.reduce((a, b) => a + b, 0) / nums.length
}

function stdev(nums) {
  const m = mean(nums)
  return Math.sqrt(mean(nums.map((n) => (n - m) ** 2)))
}

function pct(n) {
  return `${(n * 100).toFixed(1)}%`
}

// Flat palette: Pure Black, Soft Ivory, with Lime as a clean accent
const ACCENT_LIME = '#B0E454'
const COLOR_IVORY = '#EAFFD0'
const COLOR_IVORY_DIM = 'rgba(234, 255, 208, 0.82)'
const RULE_STROKE = 'rgba(255, 255, 255, 0.1)'

const tooltipStyle = {
  background: '#111111',
  border: '1px solid rgba(176, 228, 84, 0.4)',
  borderRadius: 8,
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.9)',
  fontFamily: '"JetBrains Mono", ui-monospace, monospace',
  fontSize: 12,
  color: COLOR_IVORY,
  padding: '8px 12px',
}

const tickStyle = {
  fontFamily: '"JetBrains Mono", ui-monospace, monospace',
  fontSize: 11,
  fill: COLOR_IVORY_DIM,
}

function StatBand({ data }) {
  const proposedAcc = data.seed_variance['Proposed Method'].map((r) => r.accuracy)
  const baselineAcc = data.seed_variance['Baseline Model'].map((r) => r.accuracy)
  const proposedMean = mean(proposedAcc)
  const baselineMean = mean(baselineAcc)
  const deltaPp = (proposedMean - baselineMean) * 100
  const ds = data.dataset

  return (
    <div className="stat-command-deck">
      <div className="stat-deck-grid">
        <div className="stat-tile hero-tile">
          <div className="tile-glow-dot" />
          <span className="stat-label">Ensemble Accuracy</span>
          <div className="stat-metric-row">
            <span className="stat-value lime-accent">
              {pct(proposedMean)}
            </span>
            <span className="stat-sub">±{(stdev(proposedAcc) * 100).toFixed(1)}%</span>
          </div>
          <div className="stat-badge lime-badge">
            {deltaPp >= 0 ? '+' : ''}{deltaPp.toFixed(1)} pp vs WWR Baseline
          </div>
        </div>

        <div className="stat-tile">
          <span className="stat-label">Baseline Model (WWR)</span>
          <div className="stat-metric-row">
            <span className="stat-value">{pct(baselineMean)}</span>
            <span className="stat-sub">±{(stdev(baselineAcc) * 100).toFixed(1)}%</span>
          </div>
          <span className="tile-caption">FIFA World Cup history only</span>
        </div>

        <div className="stat-tile">
          <span className="stat-label">Matches Modeled</span>
          <div className="stat-metric-row">
            <span className="stat-value">{ds.n_train_matches + ds.n_test_matches}</span>
          </div>
          <span className="tile-caption">{ds.n_train_matches} train / {ds.n_test_matches} test</span>
        </div>

        <div className="stat-tile">
          <span className="stat-label">PCA Compression</span>
          <div className="stat-metric-row">
            <span className="stat-value lime-accent">{ds.n_pca_components}</span>
            <span className="stat-sub">/ {ds.n_raw_features} features</span>
          </div>
          <span className="tile-caption">&gt;=95% explained variance</span>
        </div>

        <div className="stat-tile">
          <span className="stat-label">Team Profiles</span>
          <div className="stat-metric-row">
            <span className="stat-value">{ds.n_team_profiles}</span>
          </div>
          <span className="tile-caption">189 squads (2015–2025)</span>
        </div>
      </div>

      <div className="model-roster-bar">
        <div className="roster-lead">
          <span className="roster-indicator">5-Model Ensemble</span>
          <span className="roster-names">{MODELS.join(' • ')}</span>
        </div>
        <div className="roster-meta">
          <span>Majority Vote Consensus</span>
          <span className="meta-bullet">•</span>
          <span>{data.seeds.length} Random Seeds ({data.seeds.join(', ')})</span>
        </div>
      </div>
    </div>
  )
}

function Scope({ children }) {
  return <span className="panel-scope">{children}</span>
}

function ConfusionTable({ title, matrix }) {
  const max = Math.max(...matrix.flat())
  const [tl, tr, bl, br] = matrix.flat()

  const getCellShade = (val) => {
    const ratio = max > 0 ? val / max : 0
    if (ratio > 0.65) {
      return { background: '#254a14', color: '#B0E454', borderColor: '#4a8229', fontWeight: '700' }
    }
    if (ratio > 0.35) {
      return { background: '#19330e', color: '#EAFFD0', borderColor: '#2d581a', fontWeight: '600' }
    }
    if (ratio > 0.15) {
      return { background: '#172014', color: '#c5d9b5', borderColor: 'rgba(234, 255, 208, 0.2)' }
    }
    return { background: '#131313', color: '#889a7e', borderColor: 'rgba(255, 255, 255, 0.1)' }
  }

  return (
    <div className="confusion-matrix-card">
      <h4>{title}</h4>
      <div className="confusion-grid">
        <span className="confusion-corner" />
        <span className="confusion-head">pred away</span>
        <span className="confusion-head">pred home</span>
        <span className="confusion-head confusion-head-row">act away</span>
        <div className="confusion-cell" style={getCellShade(tl)}>{tl}</div>
        <div className="confusion-cell" style={getCellShade(tr)}>{tr}</div>
        <span className="confusion-head confusion-head-row">act home</span>
        <div className="confusion-cell" style={getCellShade(bl)}>{bl}</div>
        <div className="confusion-cell" style={getCellShade(br)}>{br}</div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    let cancelled = false
    withRetry(getEvaluation, { onWait: () => !cancelled && setStarting(true) })
      .then((d) => { if (!cancelled) { setStarting(false); setData(d) } })
      .catch((err) => { if (!cancelled) { setStarting(false); setError(err.message) } })
    return () => { cancelled = true }
  }, [])

  if (error) return <p className="glass-error">{error}</p>
  if (starting) {
    return (
      <div className="glass-card loading-card">
        <div className="loading-pulse" />
        <p className="hint">
          Waiting for the backend… training the 5-seed ensemble on first start.
          This page will load automatically once ready.
        </p>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="glass-card loading-card">
        <div className="loading-pulse" />
        <p className="hint">Loading evaluation analytics…</p>
      </div>
    )
  }

  const accuracyRows = data.comparison.map((row) => ({
    metric: row['Evaluation Metric'],
    Overall: row['Overall Accuracy'],
    'High-scoring': row['Accuracy (High-scoring)'],
    'Low-scoring': row['Accuracy (Low-scoring)'],
  }))

  const seeds = data.seed_variance['Proposed Method'].map((r) => r.seed)
  const seedRows = seeds.map((seed, i) => ({
    seed: `Seed ${seed}`,
    Proposed: data.seed_variance['Proposed Method'][i].accuracy,
    Baseline: data.seed_variance['Baseline Model'][i].accuracy,
  }))

  const pcaRows = data.pca_cumulative_variance
    ? data.pca_cumulative_variance.map((v, i) => ({ component: i + 1, variance: v }))
    : null

  const perModelRows = Object.entries(data.per_model_accuracy)
    .map(([model, accuracy]) => ({ model: MODEL_LABEL[model] ?? model, accuracy }))
    .sort((a, b) => a.accuracy - b.accuracy)

  const featureImportanceRows = data.feature_importance
    ? Object.entries(data.feature_importance)
        .map(([feature, importance]) => ({ feature: humanizeFeature(feature), importance }))
        .sort((a, b) => a.importance - b.importance)
        .slice(-12)
    : null

  const labelStyle = {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fill: 'rgba(234, 255, 208, 0.95)',
  }
  const fmtLabel = (v) => `${(v * 100).toFixed(1)}%`

  return (
    <div className="dashboard-container">
      {/* 1. Executive Metric Command Deck */}
      <StatBand data={data} />

      {/* 2. Seamless Bento Grid Layout (No tab switcher) */}
      <div className="bento-dashboard-grid">
        {/* Row 1: Decision Accuracy & Confusion Matrices */}
        <section className="glass-card bento-span-7">
          <div className="card-header-bar">
            <h3>Match Decision Accuracy <Scope>seed {data.default_seed}</Scope></h3>
            <span className="card-pill">Proposed vs Baseline</span>
          </div>
          <p className="card-desc">
            Overall accuracy split by high-scoring (&gt;2.5 goals) and low-scoring matches.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={accuracyRows} margin={{ top: 25, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
              <XAxis dataKey="metric" tick={tickStyle} stroke={RULE_STROKE} />
              <YAxis domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={RULE_STROKE} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }} />
              <Bar dataKey="Overall" fill={ACCENT_LIME} radius={[3, 3, 0, 0]}>
                <LabelList dataKey="Overall" position="top" formatter={fmtLabel} style={labelStyle} />
              </Bar>
              <Bar dataKey="High-scoring" fill="#4a5a3a" radius={[3, 3, 0, 0]}>
                <LabelList dataKey="High-scoring" position="top" formatter={fmtLabel} style={labelStyle} />
              </Bar>
              <Bar dataKey="Low-scoring" fill="#2d3824" radius={[3, 3, 0, 0]}>
                <LabelList dataKey="Low-scoring" position="top" formatter={fmtLabel} style={labelStyle} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="glass-card bento-span-5 confusion-panel">
          <div className="card-header-bar">
            <h3>Classification Heatmaps <Scope>seed {data.default_seed}</Scope></h3>
            <span className="card-pill-outline">Error Analysis</span>
          </div>
          <p className="card-desc">Away win vs Home win distribution with intensity shades</p>
          <div className="confusion-pair">
            <ConfusionTable title="Proposed Ensemble" matrix={data.confusion_matrix.proposed} />
            <ConfusionTable title="WWR Baseline" matrix={data.confusion_matrix.baseline} />
          </div>
        </section>

        {/* Row 2: Stability across 5 Seeds & PCA Variance */}
        <section className="glass-card bento-span-7">
          <div className="card-header-bar">
            <h3>Ensemble Stability across 5 Seeds</h3>
            <span className="card-pill">Cross-Seed Robustness</span>
          </div>
          <p className="card-desc">Comparing generalization across randomized splits {data.seeds.join(', ')}</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={seedRows} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
              <XAxis dataKey="seed" tick={tickStyle} stroke={RULE_STROKE} />
              <YAxis domain={['dataMin - 0.04', 'dataMax + 0.04']} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={RULE_STROKE} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase' }} />
              <Line type="monotone" dataKey="Proposed" stroke={ACCENT_LIME} strokeWidth={2.5} dot={{ r: 4, fill: '#000000', stroke: ACCENT_LIME, strokeWidth: 2 }} activeDot={{ r: 5 }} />
              <Line type="monotone" dataKey="Baseline" stroke={COLOR_IVORY_DIM} strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3, fill: '#000000', stroke: COLOR_IVORY_DIM }} />
            </LineChart>
          </ResponsiveContainer>
        </section>

        {pcaRows && (
          <section className="glass-card bento-span-5">
            <div className="card-header-bar">
              <h3>PCA Variance Explained <Scope>seed {data.default_seed}</Scope></h3>
              <span className="card-pill-outline">{pcaRows.length} Components</span>
            </div>
            <p className="card-desc">Cumulative information retained from raw player attributes</p>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={pcaRows} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
                <XAxis dataKey="component" tick={tickStyle} stroke={RULE_STROKE} />
                <YAxis domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={RULE_STROKE} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="variance" stroke={ACCENT_LIME} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </section>
        )}

        {/* Row 3: Per-Model Rankings & ROC Curve */}
        <section className="glass-card bento-span-6">
          <div className="card-header-bar">
            <h3>Individual Model Rankings <Scope>pre-vote</Scope></h3>
            <span className="card-pill">5 Architectures</span>
          </div>
          <p className="card-desc">Performance of each classifier before majority-vote aggregation</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perModelRows} layout="vertical" margin={{ top: 5, right: 40, left: 15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
              <XAxis type="number" domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={RULE_STROKE} />
              <YAxis type="category" dataKey="model" width={130} tick={tickStyle} stroke={RULE_STROKE} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Bar dataKey="accuracy" fill="#445533" stroke={ACCENT_LIME} strokeWidth={1} radius={[0, 3, 3, 0]}>
                <LabelList dataKey="accuracy" position="right" formatter={fmtLabel} style={labelStyle} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="glass-card bento-span-6">
          <div className="card-header-bar">
            <h3>ROC Discrimination Curve <Scope>AUC = {data.roc.auc.toFixed(3)}</Scope></h3>
            <span className="card-pill lime-badge">High Separability</span>
          </div>
          <p className="card-desc">True Positive Rate vs False Positive Rate across decision thresholds</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.roc.points} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={tickStyle} stroke={RULE_STROKE} />
              <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={tickStyle} stroke={RULE_STROKE} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `FPR: ${v.toFixed(3)}`} />
              <Line type="monotone" dataKey="tpr" stroke={ACCENT_LIME} dot={false} strokeWidth={2.5} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>

        {/* Row 4: Top Feature Drivers */}
        {featureImportanceRows && (
          <section className="glass-card bento-span-12">
            <div className="card-header-bar">
              <h3>Dominant Feature Drivers <Scope>mapped through PCA back to real attributes</Scope></h3>
              <span className="card-pill">RF + XGBoost Importance</span>
            </div>
            <p className="card-desc">
              Highest-weighted relative squad attributes driving match outcome predictions.
            </p>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={featureImportanceRows} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={RULE_STROKE} />
                <XAxis type="number" tick={tickStyle} stroke={RULE_STROKE} />
                <YAxis type="category" dataKey="feature" width={180} tick={tickStyle} stroke={RULE_STROKE} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="importance" fill={ACCENT_LIME} radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </section>
        )}

        {/* Row 5: Scoreline Model Engine */}
        {data.score_model && (
          <section className="glass-card bento-span-12">
            <div className="card-header-bar">
              <h3>Dixon-Coles & Poisson Scoreline Simulation Engine</h3>
              <span className="card-pill">Goal Distribution Model</span>
            </div>
            <p className="card-desc">
              Bivariate Poisson model with Dixon-Coles low-scoreline dependency correction (ρ = {data.score_model.dixon_coles_rho.toFixed(4)}).
              Predicts goals rather than binary outcomes, unlocking accurate draw probabilities.
            </p>
            <div className="score-table-wrap">
              <table className="score-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Ranked Probability Score (RPS) ↓</th>
                    <th>Log Loss ↓</th>
                    <th>Derived Win Accuracy ↑</th>
                    <th>Exact Scoreline ↑</th>
                    <th>Predicted Draw %</th>
                  </tr>
                </thead>
                <tbody>
                  {data.score_model.comparison.map((r) => (
                    <tr key={r.model} className={r.model.startsWith('Baseline') ? 'is-floor' : ''}>
                      <td className="model-cell">
                        <span className="cell-bullet" />
                        {r.model}
                      </td>
                      <td><code>{r.RPS.toFixed(4)}</code></td>
                      <td><code>{r['Log Loss'].toFixed(4)}</code></td>
                      <td><span className="badge-stat">{pct(r['Derived Win Accuracy'])}</span></td>
                      <td><span className="badge-stat">{pct(r['Exact Scoreline'])}</span></td>
                      <td>{pct(r['Predicted Draw Rate'])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="scoreline-footer-bar">
              <span>Actual Draw Rate: <strong>{pct(data.score_model.comparison[0]['Actual Draw Rate'])}</strong></span>
              <span className="meta-bullet">•</span>
              <span>Dixon-Coles Correlation Parameter: <strong>ρ = {data.score_model.dixon_coles_rho.toFixed(4)}</strong></span>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}
