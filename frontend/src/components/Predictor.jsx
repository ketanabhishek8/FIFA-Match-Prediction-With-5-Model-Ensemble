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
    <div className="card">
      <form className="predict-form" onSubmit={handlePredict}>
        <TeamCombobox
          label="Team A"
          teams={teams}
          value={teamA}
          onChange={setTeamA}
          excluded={teamB}
          excludedNote="picked as Team B"
        />
        <div className="vs">vs</div>
        <TeamCombobox
          label="Team B"
          teams={teams}
          value={teamB}
          onChange={setTeamB}
          excluded={teamA}
          excludedNote="picked as Team A"
        />
        <div className="field">
          <label htmlFor="edition-year">FIFA edition year</label>
          <select id="edition-year" value={year} onChange={(e) => setYear(e.target.value)} disabled={!years.length}>
            {years.map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <button type="submit" disabled={loading || !years.length || teamA === teamB}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      <p className="field-note">
        The ensemble is trained on every World Cup match from 2015–2025 at once — this only
        picks which edition’s squad ratings describe the two teams. Latest shared edition by default.
      </p>

      <label className="venue-toggle">
        <input type="checkbox" checked={neutral} onChange={(e) => setNeutral(e.target.checked)} />
        <span>
          Neutral venue
          <span className="venue-note">
            {neutral
              ? ' — scored both ways and averaged, so team order doesn’t change the result (World Cup default)'
              : ' — off: Team A is treated as the home side, which the model favours (62.8% of non-neutral matches are home wins)'}
          </span>
        </span>
      </label>

      {teamA && teamB && years.length > 0 && (
        <div className="request-preview">
          <span className="prompt">&gt;</span> predict(team_a=<span className="arg">"{teamA}"</span>, team_b=<span className="arg">"{teamB}"</span>, year=<span className="arg">{year}</span>)
        </div>
      )}

      {starting && (
        <p className="hint">Waiting for the backend to finish starting up… retrying automatically.</p>
      )}
      {!starting && !years.length && teamA && teamB && teamA !== teamB && (
        <p className="hint">No overlapping FIFA edition year for these two teams.</p>
      )}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className="result-head">
            <span className="result-status">predicted</span>
            <span className="result-matchup">
              {result.team_a} vs {result.team_b} · FIFA {String(result.year).slice(-2)} squad ratings
              {result.neutral ? ' · neutral venue' : ` · ${result.team_a} at home`}
            </span>
          </div>
          <div className="winner">
            <span className="trophy">&gt;</span>{result.winner} wins
          </div>
          {(() => {
            const votes = Object.values(result.model_votes)
            const total = votes.length
            const agree = votes.filter((v) => v === result.winner).length
            const loser = result.winner === result.team_a ? result.team_b : result.team_a
            const winnerPct = Math.round(result.confidence * 100)
            const loserPct = 100 - winnerPct
            return (
              <div className="consensus">
                <div className="consensus-value">
                  {winnerPct}<span className="unit">%</span>
                  <span className="sub">predicted probability · {agree}/{total} models agree</span>
                </div>
                <div className="consensus-bar">
                  <div className="consensus-bar-fill" style={{ transform: `scaleX(${result.confidence})` }} />
                </div>
                <div className="consensus-legend">
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
              <div className="scoreline">
                <h4>Scoreline</h4>
                <div className="scoreline-main">
                  <div className="scoreline-score">
                    {top.team_a}<span className="sl-dash">–</span>{top.team_b}
                  </div>
                  <div className="scoreline-meta">
                    <span className="sl-pct">{topPct}%</span> most likely
                    <span className="sl-note">
                      Football scorelines are genuinely uncertain — even the best guess is a
                      long way from a safe bet. The three-way split below is the firmer answer.
                    </span>
                  </div>
                </div>

                <div className="xg-row">
                  <span className="xg-side">{result.team_a} <strong>{s.expected_goals.team_a.toFixed(2)}</strong></span>
                  <span className="xg-label">expected goals</span>
                  <span className="xg-side"><strong>{s.expected_goals.team_b.toFixed(2)}</strong> {result.team_b}</span>
                </div>

                <div className="outcome-bar" role="img"
                     aria-label={`${result.team_a} ${pctA}%, draw ${pctDraw}%, ${result.team_b} ${pctB}%`}>
                  <div className="seg seg-a" style={{ width: `${pctA}%` }} />
                  <div className="seg seg-draw" style={{ width: `${pctDraw}%` }} />
                  <div className="seg seg-b" style={{ width: `${pctB}%` }} />
                </div>
                <div className="outcome-legend">
                  <span><strong>{pctA}%</strong> {result.team_a}</span>
                  {/* The draw is the whole point of this model: the ensemble
                      has no draw class and cannot express this number at all. */}
                  <span><strong>{pctDraw}%</strong> draw</span>
                  <span><strong>{pctB}%</strong> {result.team_b}</span>
                </div>

                <div className="alt-scores">
                  {s.top_scorelines.map((t) => (
                    <span className="alt-score" key={`${t.team_a}-${t.team_b}`}>
                      {t.team_a}–{t.team_b}
                      <span className="alt-pct">{Math.round(t.probability * 100)}%</span>
                    </span>
                  ))}
                </div>

                {!s.agrees_with_ensemble && (
                  <p className="scoreline-disagree">
                    Heads up: the scoreline model favours <strong>{s.favours}</strong>, while the
                    ensemble picks <strong>{result.winner}</strong>. They are fit to different
                    targets — goals scored versus who won — so they can disagree on close matches.
                  </p>
                )}
              </div>
            )
          })()}

          <div className="votes">
            <h4>Model votes</h4>
            <ul>
              {Object.entries(result.model_votes).map(([model, pick]) => {
                const pct = Math.round(result.model_confidence[model] * 100)
                return (
                  <li key={model}>
                    <span className="model-name">{MODEL_LABEL[model] ?? model}</span>
                    <span className="model-pick">{pick} <span className="model-pct">{pct}%</span></span>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
