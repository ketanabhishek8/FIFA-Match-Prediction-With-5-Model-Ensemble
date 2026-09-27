import Dashboard from './components/Dashboard'
import Predictor from './components/Predictor'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <span className="wordmark">FIFA Match Predictor<span className="dot">.</span></span>
        <span className="meta">5-model ensemble vs. WWR baseline</span>
      </header>

      <main>
        <Predictor />
        <Dashboard />
      </main>
    </div>
  )
}

export default App
