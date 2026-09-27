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

const ACCENT = 'oklch(78% 0.075 80)'
const ACCENT_SOFT = 'oklch(78% 0.075 80 / 0.6)'
const ACCENT_FAINT = 'oklch(78% 0.075 80 / 0.32)'
const MUTED = 'oklch(64% 0.016 77)'

const tooltipStyle = {
  background: 'oklch(9% 0.007 75)',
  border: '1px solid oklch(32% 0.014 75)',
  borderRadius: 10,
  fontFamily: '"JetBrains Mono", ui-monospace, monospace',
  fontSize: 12,
  color: '#fff',
}
const tickStyle = { fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: 11, fill: 'oklch(64% 0.016 77)' }

function StatBand({ data }) {
  const proposedAcc = data.seed_variance['Proposed Method'].map((r) => r.accuracy)
  const baselineAcc = data.seed_variance['Baseline Model'].map((r) => r.accuracy)
  const proposedMean = mean(proposedAcc)
  const baselineMean = mean(baselineAcc)
  const deltaPp = (proposedMean - baselineMean) * 100
  const ds = data.dataset

  return (
    <div className="stat-band">
      <div className="stat-grid">
        <div className="stat-tile">
          <span className="stat-value accent">{pct(proposedMean)}<span className="sub">±{(stdev(proposedAcc) * 100).toFixed(1)}</span></span>
          <span className="stat-label">Ensemble accuracy</span>
        </div>
        <div className="stat-tile">
          <span className="stat-value">{pct(baselineMean)}<span className="sub">±{(stdev(baselineAcc) * 100).toFixed(1)}</span></span>
          <span className="stat-label">Baseline (WWR)</span>
        </div>
        <div className="stat-tile">
          <span className="stat-value accent">{deltaPp >= 0 ? '+' : ''}{deltaPp.toFixed(1)}<span className="sub">pp</span></span>
          <span className="stat-label">Improvement</span>
        </div>
        <div className="stat-tile">
          <span className="stat-value">{ds.n_train_matches + ds.n_test_matches}</span>
          <span className="stat-label">Matches modeled</span>
        </div>
        <div className="stat-tile">
          <span className="stat-value">{ds.n_team_profiles}</span>
          <span className="stat-label">Team-year profiles</span>
        </div>
        <div className="stat-tile">
          <span className="stat-value">{ds.n_pca_components}<span className="sub">/{ds.n_raw_features}</span></span>
          <span className="stat-label">PCA components</span>
        </div>
      </div>
      <div className="model-roster">
        <strong>5-model ensemble</strong> — {MODELS.join(' · ')} · majority vote
        <span className="panel-scope">
          accuracy figures averaged over {data.seeds.length} seeds ({data.seeds.join(', ')})
        </span>
      </div>
    </div>
  )
}

// Most panels are computed from a single seed's held-out test split, while the
// stat tiles and the stability chart average across all seeds. Without a label
// the page looks self-contradictory: the tiles can show the ensemble ahead
// while the single-seed comparison below shows the baseline winning.
function Scope({ children }) {
  return <span className="panel-scope">{children}</span>
}

function ConfusionTable({ title, matrix }) {
  const max = Math.max(...matrix.flat())
  const [tl, tr, bl, br] = matrix.flat()
  return (
    <div className="confusion">
      <h4>{title}</h4>
      <div className="confusion-grid">
        <span className="confusion-corner" />
        <span className="confusion-head">predicted away</span>
        <span className="confusion-head">predicted home</span>
        <span className="confusion-head confusion-head-row">actual away</span>
        <div className="confusion-cell" style={{ '--intensity': tl / max }}>{tl}</div>
        <div className="confusion-cell" style={{ '--intensity': tr / max }}>{tr}</div>
        <span className="confusion-head confusion-head-row">actual home</span>
        <div className="confusion-cell" style={{ '--intensity': bl / max }}>{bl}</div>
        <div className="confusion-cell" style={{ '--intensity': br / max }}>{br}</div>
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

  if (error) return <p className="error">{error}</p>
  if (starting) {
    return (
      <p className="hint">
        Waiting for the backend… it trains the 5-seed ensemble on first start,
        which takes a few minutes. This page will load itself when it's ready.
      </p>
    )
  }
  if (!data) return <p className="hint">Loading evaluation results…</p>

  const accuracyRows = data.comparison.map((row) => ({
    metric: row['Evaluation Metric'],
    Overall: row['Overall Accuracy'],
    'High-scoring': row['Accuracy (High-scoring)'],
    'Low-scoring': row['Accuracy (Low-scoring)'],
  }))

  const seeds = data.seed_variance['Proposed Method'].map((r) => r.seed)
  const seedRows = seeds.map((seed, i) => ({
    seed,
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

  const labelStyle = { fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--color-ink-muted)' }
  const fmtLabel = (v) => `${(v * 100).toFixed(1)}%`

  return (
    <div className="dashboard">
      <StatBand data={data} />

      <section className="card">
        <h3>Accuracy: proposed vs. baseline <Scope>seed {data.default_seed} only</Scope></h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={accuracyRows} margin={{ top: 20, right: 0, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
            <XAxis dataKey="metric" tick={tickStyle} />
            <YAxis domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }} />
            <Bar dataKey="Overall" fill={ACCENT} radius={[3, 3, 0, 0]}>
              <LabelList dataKey="Overall" position="top" formatter={fmtLabel} style={labelStyle} />
            </Bar>
            <Bar dataKey="High-scoring" fill={ACCENT_SOFT} radius={[3, 3, 0, 0]}>
              <LabelList dataKey="High-scoring" position="top" formatter={fmtLabel} style={labelStyle} />
            </Bar>
            <Bar dataKey="Low-scoring" fill={ACCENT_FAINT} radius={[3, 3, 0, 0]}>
              <LabelList dataKey="Low-scoring" position="top" formatter={fmtLabel} style={labelStyle} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section className="result confusion-panel">
        <div className="result-head">
          <span className="result-status">error analysis</span>
          <span className="result-matchup">
            predicted: away win → home win <Scope>seed {data.default_seed} only</Scope>
          </span>
        </div>
        <div className="confusion-pair">
          <ConfusionTable title="Proposed" matrix={data.confusion_matrix.proposed} />
          <ConfusionTable title="Baseline" matrix={data.confusion_matrix.baseline} />
        </div>
      </section>

      <div className="chart-row">
        <section className="card">
          <h3>Stability across 5 seeds <Scope>all seeds: {data.seeds.join(', ')}</Scope></h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={seedRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
              <XAxis dataKey="seed" tick={tickStyle} />
              <YAxis domain={['dataMin - 0.03', 'dataMax + 0.03']} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }} />
              <Line type="monotone" dataKey="Proposed" stroke={ACCENT} strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="Baseline" stroke={MUTED} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </section>

        {pcaRows && (
          <section className="card">
            <h3>PCA cumulative variance <Scope>seed {data.default_seed} only</Scope></h3>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={pcaRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
                <XAxis dataKey="component" tick={tickStyle} label={{ value: 'components', position: 'insideBottom', offset: -5, style: tickStyle }} />
                <YAxis domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="variance" stroke={ACCENT} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </section>
        )}
      </div>

      <div className="chart-row">
        <section className="card">
          <h3>Per-model accuracy (before majority vote) <Scope>seed {data.default_seed} only</Scope></h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perModelRows} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
              <XAxis type="number" domain={[0, 1]} tick={tickStyle} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <YAxis type="category" dataKey="model" width={130} tick={tickStyle} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Bar dataKey="accuracy" fill={ACCENT} radius={[0, 3, 3, 0]}>
                <LabelList dataKey="accuracy" position="right" formatter={fmtLabel} style={labelStyle} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="card">
          <h3>ROC curve (ensemble, AUC={data.roc.auc.toFixed(3)}) <Scope>seed {data.default_seed} only</Scope></h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.roc.points} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={tickStyle}
                label={{ value: 'false positive rate', position: 'insideBottom', offset: -5, style: tickStyle }} />
              <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={tickStyle} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `fpr ${v.toFixed(3)}`} />
              <Line type="monotone" dataKey="tpr" stroke={ACCENT} dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>

      {data.score_model && (
        <section className="card">
          <h3>Scoreline model <Scope>seed {data.default_seed} only</Scope></h3>
          <p className="score-panel-note">
            Predicts goals rather than a winner, so it also yields a draw probability — which the
            binary ensemble structurally cannot express. Shown as a table rather than a chart on
            purpose: the Dixon-Coles correction moves RPS by about 0.0001, and bars that size
            would read as identical.
          </p>
          <div className="score-table-wrap">
            <table className="score-table">
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>RPS ↓</th>
                  <th>Log loss ↓</th>
                  <th>Win acc ↑</th>
                  <th>Exact ↑</th>
                  <th>Pred. draw</th>
                </tr>
              </thead>
              <tbody>
                {data.score_model.comparison.map((r) => (
                  <tr key={r.model} className={r.model.startsWith('Baseline') ? 'is-floor' : ''}>
                    <td>{r.model}</td>
                    <td>{r.RPS.toFixed(4)}</td>
                    <td>{r['Log Loss'].toFixed(4)}</td>
                    <td>{pct(r['Derived Win Accuracy'])}</td>
                    <td>{pct(r['Exact Scoreline'])}</td>
                    <td>{pct(r['Predicted Draw Rate'])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="score-panel-note">
            Actual draw rate <strong>{pct(data.score_model.comparison[0]['Actual Draw Rate'])}</strong>
            {' · '}Dixon-Coles ρ = <strong>{data.score_model.dixon_coles_rho.toFixed(4)}</strong>
            {' · '}win accuracy is measured on decided matches only, so it is comparable to the
            ensemble above; exact scoreline is only meaningful against its own floor.
          </p>
        </section>
      )}

      {featureImportanceRows && (
        <section className="card">
          <h3>Top features driving predictions <Scope>seed {data.default_seed} only</Scope></h3>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={featureImportanceRows} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
              <XAxis type="number" tick={tickStyle} />
              <YAxis type="category" dataKey="feature" width={170} tick={tickStyle} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="importance" fill={ACCENT_SOFT} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}
    </div>
  )
}
