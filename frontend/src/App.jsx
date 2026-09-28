import { lazy, Suspense, useState } from 'react'
import Dashboard from './components/Dashboard'
import Predictor from './components/Predictor'
import './App.css'

const TrophyScene = lazy(() => import('./components/TrophyScene'))

function App() {
  const [focusTrophy, setFocusTrophy] = useState(false)

  return (
    <>
      <Suspense fallback={null}>
        <TrophyScene />
      </Suspense>
      <div className="app">
        <header className="app-header">
          <div className="brand-group">
            <div className="status-pill">
              <span className="pulse-dot" />
              <span className="status-text">LIVE INFERENCE ENGINE</span>
            </div>
            <h1 className="wordmark">
              FIFA Match Predictor<span className="dot">.</span>
            </h1>
            <p className="sub-tagline">
              5-model ensemble + Dixon-Coles scoreline model vs. WWR baseline
            </p>
          </div>

          <div className="header-meta-chips">
            <span className="chip">Trained 2015–2024 · 2025 held out</span>
            <span className="chip">189 National Squads</span>
            <span className="chip accent-chip">5 Seeds Evaluated</span>
            <button
              type="button"
              className={`chip trophy-toggle-btn ${focusTrophy ? 'active-toggle' : ''}`}
              aria-pressed={focusTrophy}
              onClick={() => setFocusTrophy(!focusTrophy)}
            >
              {focusTrophy ? 'Back to Dashboard' : 'Trophy Room'}
            </button>
          </div>
        </header>

        <main className={`app-main ${focusTrophy ? 'trophy-focused' : ''}`} inert={focusTrophy}>
          <Predictor />
          <Dashboard />
        </main>
      </div>
    </>
  )
}

export default App
