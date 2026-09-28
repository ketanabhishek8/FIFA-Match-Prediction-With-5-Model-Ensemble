import { useEffect, useMemo, useState } from 'react'
import { getTeams, predict, withRetry } from '../api'
import TeamCombobox from './TeamCombobox'
import { MODEL_LABEL } from '../constants'

const DEFAULT_MATCHUP = ['Brazil', 'Germany']

export default function Predictor() {
  const [teams, setTeams] = useState([])
  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')
  const [year, setYear] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [neutral, setNeutral] = useState(true)

  useEffect(() => {
    let cancelled = false
    withRetry(getTeams, { onWait: () => !cancelled && setStarting(true) })
      .then((data) => {
        if (cancelled) return
        setStarting(false)
        setTeams(data)
        const names = new Set(data.map((t) => t.team))
        const [a, b] = DEFAULT_MATCHUP.every((n) => names.has(n))
          ? DEFAULT_MATCHUP
          : data.slice(0, 2).map((t) => t.team)
        if (a && b) {
          setTeamA(a)
          setTeamB(b)
        }
      })
      .catch((err) => {
        if (cancelled) return
        setStarting(false)
        setError(err.message)
      })
    return () => { cancelled = true }
  }, [])

  const years = useMemo(() => {
    const a = teams.find((t) => t.team === teamA)
    const b = teams.find((t) => t.team === teamB)
    if (!a || !b) return []
    return a.years.filter((y) => b.years.includes(y))
  }, [teams, teamA, teamB])

  useEffect(() => {
    if (years.length && !years.includes(Number(year))) setYear(years[years.length - 1])
  }, [years]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handlePredict(e) {
    e.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const data = await predict(teamA, teamB, Number(year), neutral)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="glass-card predictor-arena">
      <div className="card-header-bar">
        <div>
          <h3>Match Simulation Arena</h3>
          <p className="card-desc">Simulate head-to-head fixtures using edition-specific national team ratings</p>
        </div>
        <div className="arena-badge">
          <span className="badge-pulse" />
          <span>5-Model Ensemble</span>
        </div>
      </div>

      <form className="predict-form" onSubmit={handlePredict}>
        <div className="team-select-wrapper">
          <TeamCombobox
            label="Team A"
            teams={teams}
            value={teamA}
            onChange={setTeamA}
            excluded={teamB}
            excludedNote="picked as Team B"
          />
        </div>

        <div className="vs-emblem">
          <span className="vs-text">VS</span>
        </div>

        <div className="team-select-wrapper">
          <TeamCombobox
            label="Team B"
            teams={teams}
            value={teamB}
            onChange={setTeamB}
            excluded={teamA}
            excludedNote="picked as Team A"
          />
        </div>

        <div className="controls-row">
          <div className="field-group">
            <label htmlFor="edition-year">FIFA Edition Roster</label>
            <div className="select-container">
              <select
                id="edition-year"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                disabled={!years.length}
              >
                {years.map((y) => (
                  <option key={y} value={y}>
                    FIFA {String(y).slice(-2)} ({y})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <label className="venue-toggle-switch">
            <input
              type="checkbox"
              checked={neutral}
              onChange={(e) => setNeutral(e.target.checked)}
            />
            <span className="switch-slider" />
            <span className="switch-text">
              <strong>Neutral Venue</strong>
              <span className="switch-sub">
                {neutral
                  ? 'Scored both ways and averaged, so team order doesn’t matter (World Cup default)'
                  : 'Team A is the home side, which the model favours (62.8% of non-neutral matches are home wins)'}
              </span>
            </span>
          </label>

          <button
            type="submit"
            className="predict-cta"
            disabled={loading || !years.length || teamA === teamB}
          >
            {loading ? (
              <span className="btn-inner">
                <span className="spinner" /> Calculating…
              </span>
            ) : (
              <span className="btn-inner">
                <span>PREDICT MATCH</span>
                <span className="btn-arrow">→</span>
              </span>
            )}
          </button>
        </div>
      </form>

      <p className="card-desc">
        The ensemble is trained on World Cup matches from 2015–2024 (2025 is held out for
        testing) — the edition year only picks which squad ratings describe the two teams.
        Latest shared edition by default.
      </p>

      {teamA && teamB && years.length > 0 && (
        <div className="request-terminal-badge">
          <span className="terminal-prompt">$</span>
          <span className="terminal-fn">predict</span>
          <span className="terminal-args">
            (team_a="<strong>{teamA}</strong>", team_b="<strong>{teamB}</strong>", year=<strong>{year}</strong>, neutral=<strong>{String(neutral)}</strong>)
          </span>
        </div>
      )}

      {starting && (
        <p className="hint">Waiting for the backend to finish starting up… retrying automatically.</p>
      )}
      {!starting && !years.length && teamA && teamB && teamA !== teamB && (
        <p className="hint">No overlapping FIFA edition year found for these two teams.</p>
      )}
      {error && <p className="glass-error">{error}</p>}

      {result && (
        <div className="glass-card result-display-card">
          <div className="result-headline-bar">
            <div className="result-meta-tags">
              <span className="status-chip success-chip">PREDICTION RESOLVED</span>
              <span className="fixture-tag">
                {result.team_a} vs {result.team_b} • FIFA {String(result.year).slice(-2)} Squads
                {result.neutral ? ' • Neutral Stadium' : ` • ${result.team_a} Home Turf`}
              </span>
            </div>
          </div>

          <div className="winner-podium">
            <div className="winner-details">
              <span className="winner-label">Projected Winner</span>
              <h2 className="winner-name">{result.winner} Wins</h2>
            </div>
          </div>

          {(() => {
            const votes = Object.values(result.model_votes)
            const total = votes.length
            const agree = votes.filter((v) => v === result.winner).length
            const loser = result.winner === result.team_a ? result.team_b : result.team_a
            const winnerPct = Math.round(result.confidence * 100)
            const loserPct = 100 - winnerPct

            return (
              <div className="consensus-display-deck">
                <div className="consensus-numbers">
                  <div className="prob-stat">
                    <span className="prob-big lime-text">{winnerPct}%</span>
                    <span className="prob-desc">Win Probability</span>
                  </div>
                  <div className="consensus-meter-text">
                    <strong>{agree} of {total}</strong> ensemble models voted for {result.winner}
                  </div>
                </div>

                <div className="prob-meter-track">
                  <div className="prob-meter-fill" style={{ width: `${result.confidence * 100}%` }} />
                </div>

                <div className="prob-meter-labels">
                  <span><strong>{winnerPct}%</strong> {result.winner}</span>
                  <span><strong>{loserPct}%</strong> {loser}</span>
                </div>
              </div>
            )
          })()}

          {result.score && (() => {
            const s = result.score
            const top = s.most_likely
            const topPct = Math.round(top.probability * 100)
            const op = s.outcome_probability
            // Round once and give the remainder to the last segment, so the
            // three numbers always add to 100 in the legend.
            const pctA = Math.round(op.team_a * 100)
            const pctDraw = Math.round(op.draw * 100)
            const pctB = 100 - pctA - pctDraw

            return (
              <div className="scoreline-deck">
                <div className="deck-header">
                  <h4>Dixon-Coles Projected Scoreline</h4>
                  <span className="xg-badge">Poisson Engine</span>
                </div>

                <div className="scoreline-showcase">
                  <div className="score-big">
                    <span>{top.team_a}</span>
                    <span className="score-dash">:</span>
                    <span>{top.team_b}</span>
                  </div>
                  <div className="score-caption">
                    <span className="score-chance-chip">{topPct}% Most Likely Exact Scoreline</span>
                    <p className="uncertainty-note">
                      Football scorelines are genuinely uncertain — even the best guess is a long way from a
                      safe bet. The three-way split below is the firmer answer.
                    </p>
                  </div>
                </div>

                <div className="xg-stats-row">
                  <div className="xg-stat-item">
                    <span className="xg-country">{result.team_a}</span>
                    <span className="xg-num lime-text">{s.expected_goals.team_a.toFixed(2)}</span>
                    <span className="xg-tag">xG Expected Goals</span>
                  </div>
                  <div className="xg-divider">VS</div>
                  <div className="xg-stat-item">
                    <span className="xg-country">{result.team_b}</span>
                    <span className="xg-num ivory-text">{s.expected_goals.team_b.toFixed(2)}</span>
                    <span className="xg-tag">xG Expected Goals</span>
                  </div>
                </div>

                <div
                  className="three-way-probability-bar"
                  role="img"
                  aria-label={`${result.team_a} ${pctA}%, draw ${pctDraw}%, ${result.team_b} ${pctB}%`}
                >
                  <div className="three-seg seg-a" style={{ width: `${pctA}%` }} title={`${result.team_a}: ${pctA}%`} />
                  <div className="three-seg seg-draw" style={{ width: `${pctDraw}%` }} title={`Draw: ${pctDraw}%`} />
                  <div className="three-seg seg-b" style={{ width: `${pctB}%` }} title={`${result.team_b}: ${pctB}%`} />
                </div>

                <div className="three-way-legend">
                  <span className="legend-item"><span className="legend-dot dot-lime" /><strong>{pctA}%</strong> {result.team_a} Win</span>
                  <span className="legend-item"><span className="legend-dot dot-forest" /><strong>{pctDraw}%</strong> Match Draw</span>
                  <span className="legend-item"><span className="legend-dot dot-ivory" /><strong>{pctB}%</strong> {result.team_b} Win</span>
                </div>

                <div className="alternative-scorelines-list">
                  <span className="alt-title">Alternative Likely Scorelines:</span>
                  <div className="alt-chips-container">
                    {s.top_scorelines.map((t) => (
                      <span className="alt-chip" key={`${t.team_a}-${t.team_b}`}>
                        <span className="alt-score-nums">{t.team_a} - {t.team_b}</span>
                        <span className="alt-score-prob">{Math.round(t.probability * 100)}%</span>
                      </span>
                    ))}
                  </div>
                </div>

                {!s.agrees_with_ensemble && (
                  <div className="model-divergence-callout">
                    <span className="notice-tag">Notice:</span>
                    <p>
                      The scoreline goal-distribution model favours <strong>{s.favours}</strong>, while the
                      majority-vote ensemble picks <strong>{result.winner}</strong>. They are fit to different
                      targets — goals scored versus who won — so they can disagree on close matches.
                    </p>
                  </div>
                )}
              </div>
            )
          })()}

          <div className="model-votes-section">
            <h4>Ensemble Model Consensus Breakdown</h4>
            <div className="votes-grid">
              {Object.entries(result.model_votes).map(([model, pick]) => {
                const pct = Math.round(result.model_confidence[model] * 100)
                const isWinner = pick === result.winner
                return (
                  <div key={model} className={`vote-card ${isWinner ? 'vote-agreed' : 'vote-dissent'}`}>
                    <span className="vote-model-name">{MODEL_LABEL[model] ?? model}</span>
                    <div className="vote-result-row">
                      <span className="vote-pick">{pick}</span>
                      <span className="vote-conf">{pct}%</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
