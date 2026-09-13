import { useState, useEffect } from 'react'

const API = 'http://127.0.0.1:8777'
const TOKEN = 'demo-shared-token'

async function api(path: string, options?: RequestInit) {
  const res = await fetch(`${API}${path}`, {
    headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
}

function Page({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-indigo-700 text-white">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          <div className="flex items-center gap-3">
            <span className="text-sm opacity-80">token: {TOKEN}</span>
          </div>
        </div>
      </header>
      <main className="max-w-5xl mx-auto px-4 py-6">{children}</main>
    </div>
  )
}

function Badge({ label, color = 'gray' }: { label: string; color?: string }) {
  const colors: Record<string, string> = {
    gray: 'bg-gray-100 text-gray-700',
    green: 'bg-green-100 text-green-800',
    red: 'bg-red-100 text-red-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    blue: 'bg-blue-100 text-blue-800',
  }
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[color]}`}>{label}</span>
}

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm ${className}`}>{children}</div>
  )
}

function Empty({ message }: { message: string }) {
  return (
    <div className="text-center py-12 text-gray-400">
      <p className="text-lg">{message}</p>
    </div>
  )
}

function Button({
  children,
  variant = 'primary',
  className = '',
  ...rest
}: {
  children: React.ReactNode
  variant?: 'primary' | 'danger' | 'ghost'
  className?: string
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    'px-3 py-1.5 rounded-md text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed'
  const variants = {
    primary: 'bg-indigo-600 text-white hover:bg-indigo-700',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    ghost: 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50',
  }
  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...rest}>
      {children}
    </button>
  )
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

  useEffect(() => { load() }, [])

  if (!state) {
    return (
      <Page title="RouteBench - Routing State">
        <Button onClick={load} disabled={loading} variant="ghost" className="mb-4">
          Refresh
        </Button>
        {error ? (
          <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200">{error}</div>
        ) : null}
        <Empty message="Click Refresh to load routing state." />
      </Page>
    )
  }

  return (
    <Page title="RouteBench - Routing State">
      <Button onClick={load} disabled={loading} variant="ghost" className="mb-4">
        Refresh
      </Button>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
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
        </Card>

        <Card>
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
              <Badge label={state.circuit_open ? 'OPEN' : 'closed'} color={state.circuit_open ? 'red' : 'green'} />
            </div>
          </div>
        </Card>
      </div>
    </Page>
  )
}

export default Routing
