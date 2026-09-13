import { useState } from 'react'
import ApprovalQueue from './pages/ApprovalQueue'
import TraceViewer from './pages/TraceViewer'
import Incidents from './pages/Incidents'
import Routing from './pages/Routing'

function App() {
  const [tab, setTab] = useState<'approvals' | 'traces' | 'incidents' | 'routing'>('approvals')

  return (
    <div className="min-h-screen">
      <nav className="bg-gray-900 text-white px-4 py-3">
        <div className="max-w-5xl mx-auto flex gap-4 text-sm font-medium">
          {([
            ['approvals', 'Approvals'],
            ['traces', 'Traces'],
            ['incidents', 'Incidents'],
            ['routing', 'Routing'],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`py-1 px-2 rounded transition ${
                tab === key
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </nav>
      <main className="max-w-5xl mx-auto px-4 py-6">
        {tab === 'approvals' && <ApprovalQueue />}
        {tab === 'traces' && <TraceViewer />}
        {tab === 'incidents' && <Incidents />}
        {tab === 'routing' && <Routing />}
      </main>
    </div>
  )
}

export default App
