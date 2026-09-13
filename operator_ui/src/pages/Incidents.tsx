import { useState } from 'react'

const API = 'http://127.0.0.1:8777'
const TOKEN = 'demo-shared-token'

async function api(path: string, options?: RequestInit) {
  const res = await fetch(`${API}${path}`, {
    headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
}

function PageHeader({ title }: { title: string }) {
  return (
    <header className="bg-indigo-700 text-white px-4 py-3 mb-6">
      <h1 className="text-lg font-semibold">{title}</h1>
    </header>
  )
}

function Empty({ msg }: { msg: string }) {
  return <div className="text-center py-12 text-gray-400">{msg}</div>
}

function Badge({ label, color }: { label: string; color: string }) {
  return <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${color}`}>{label}</span>
}

type Incident = {
  id: string
  status: string
  title: string
  started_at: string
  report_path: string | null
}

function Incidents() {
  const [items, setItems] = useState<Incident[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api('/incidents')
      setItems(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const statusColor = (s: string) => {
    if (s === 'open') return 'bg-red-100 text-red-800'
    if (s === 'resolved') return 'bg-green-100 text-green-800'
    if (s === 'monitoring') return 'bg-yellow-100 text-yellow-800'
    return 'bg-gray-100 text-gray-700'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <PageHeader title="Incident Commander" />
      <button
        onClick={load}
        disabled={loading}
        className="mb-4 px-3 py-1.5 bg-white text-gray-700 border border-gray-300 rounded-md text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
      >
        Refresh
      </button>
      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}
      {items.length === 0 ? (
        <Empty msg="No incidents recorded." />
      ) : (
        <div className="space-y-3">
          {items.map((inc) => (
            <div key={inc.id} className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Badge label={inc.status} color={statusColor(inc.status)} />
                    <span className="text-sm text-gray-400">started {new Date(inc.started_at).toLocaleString()}</span>
                  </div>
                  <h3 className="font-medium text-gray-900">{inc.title}</h3>
                  {inc.report_path ? (
                    <a
                      href={inc.report_path}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-indigo-600 hover:underline mt-1 inline-block"
                    >
                      View incident report →
                    </a>
                  ) : (
                    <p className="text-sm text-gray-400 mt-1">report pending ...</p>
                  )}
                </div>
                <Badge label={inc.id} color="bg-gray-100 text-gray-700" />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Incidents
