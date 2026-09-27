const BASE = '/api'

// The backend trains the whole pipeline during startup on a cache miss, and
// uvicorn doesn't accept connections until that finishes (several minutes).
// From the browser that looks like a dead server: the Vite dev proxy turns the
// connection refusal into a 5xx with no JSON body. Treat those as "not ready
// yet" rather than as an error, so the UI can wait instead of giving up.
const TRANSIENT_STATUS = new Set([500, 502, 503, 504])

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch {
    // Network-level failure (no proxy in front, or the dev server is down).
    throw Object.assign(new Error('Cannot reach the backend'), { transient: true })
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    // A `detail` field means FastAPI answered and is deliberately rejecting
    // this request (e.g. an unknown team) -- that's a real error, not a warm-up.
    const transient = TRANSIENT_STATUS.has(res.status) && !body.detail
    throw Object.assign(new Error(body.detail || `Request failed: ${res.status}`), {
      status: res.status,
      transient,
    })
  }
  return res.json()
}

/**
 * Run `fn`, retrying while the backend is still starting up.
 * `onWait(attempt)` fires before each wait so the caller can show a status
 * message. Non-transient failures reject immediately.
 */
export async function withRetry(fn, { onWait, attempts = 100, delayMs = 3000 } = {}) {
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn()
    } catch (err) {
      if (!err.transient) throw err
      onWait?.(attempt)
      await new Promise((resolve) => setTimeout(resolve, delayMs))
    }
  }
  throw new Error('The backend is taking longer than expected to start. Is it still running?')
}

export const getTeams = () => request('/teams')

// `neutral` averages both home/away orientations so the winner doesn't depend
// on which team was picked first -- the right default for a World Cup, which is
// played on neutral ground. See api/main.py's PredictRequest.
export const predict = (team_a, team_b, year, neutral = true) =>
  request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ team_a, team_b, year, neutral }),
  })

export const getEvaluation = () => request('/evaluation')
