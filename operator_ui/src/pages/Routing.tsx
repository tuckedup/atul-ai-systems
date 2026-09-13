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

type RoutingState = {
  model: string
  provider: string
  quality: Record<string, number>
  input_per_1m: number
  output_per_1m: number
  p95_latency_ms: number
  circuit_open: boolean
  traffic_weights: Record<string, number>
}

function Routing() {
  const [state, setState] = useState<RoutingState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api('/routing/state')
      setState(data as RoutingState)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (!state) {
    return (
      <div className="min-h-screen bg-gray-50">
        <PageHeader title="RouteBench - Routing State" />
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
        <Empty msg="Click Refresh to load routing state." />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <PageHeader title="RouteBench - Routing State" />
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
          <h3 className="font-medium text-gray-900 mb-3">Backend quality x cost matrix</h3>
          <ul className="space-y-2 text-sm">
            <li className="flex justify-between"><span className="text-gray-500">Model</span><span className="font-medium">{state.model}</span></li>
            <li className="flex justify-between"><span className="text-gray-500">Provider</span><span className="font-medium">{state.provider}</span></li>
            {Object.entries(state.quality).map(([name, q]) => (
              <li key={name} className="flex justify-between">
                <span className="text-gray-500">{name}</span>
                <span className="font-medium">{q.toFixed(3)}</span>
              </li>
            ))}
            <li className="flex justify-between border-t border-gray-200 pt-2 mt-2">
              <span className="text-gray-500">input / 1M tok</span>
              <span className="font-medium">${state.input_per_1m.toFixed(4)}</span>
            </li>
            <li className="flex justify-between">
              <span className="text-gray-500">output / 1M tok</span>
              <span className="font-medium">${state.output_per_1m.toFixed(4)}</span>
            </li>
            <li className="flex justify-between">
              <span className="text-gray-500">p95 latency</span>
              <span className="font-medium">{state.p95_latency_ms.toFixed(0)} ms</span>
            </li>
          </ul>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
          <h3 className="font-medium text-gray-900 mb-3">Traffic weights</h3>
          <ul className="space-y-2 text-sm">
            {Object.entries(state.traffic_weights).map(([model, weight]) => (
              <li key={model} className="flex justify-between">
                <span className="text-gray-500">{model}</span>
                <span className="font-medium">{weight}</span>
              </li>
            ))}
          </ul>
          <div className="mt-4 pt-3 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Circuit breaker</span>
              <Badge label={state.circuit_open ? 'OPEN' : 'closed'} color={state.circuit_open ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Routing
