import { useEffect, useState } from 'react'

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

function Button({
  children,
  variant = 'primary',
  className = '',
  loading = false,
  ...rest
}: {
  children: React.ReactNode
  variant?: 'primary' | 'danger' | 'ghost'
  className?: string
  loading?: boolean
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    'px-3 py-1.5 rounded-md text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed'
  const variants = {
    primary: 'bg-indigo-600 text-white hover:bg-indigo-700',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    ghost: 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50',
  }
  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...rest} disabled={loading || rest.disabled}>
      {children}
    </button>
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

function PageNotFound() {
  return (
    <Page title="Operator Console">
      <Empty message="Page not found." />
    </Page>
  )
}

function App() {
  const [tab, setTab] = useState<'approvals' | 'traces' | 'incidents' | 'routing'>('approvals')

  return (
    <>
      <nav className="bg-gray-100 border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 py-2 flex gap-4 text-sm font-medium">
          {([
            ['approvals', 'Approvals'],
            ['traces', 'Traces'],
            ['incidents', 'Incidents'],
            ['routing', 'Routing'],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`py-1 px-2 rounded ${tab === key ? 'bg-white shadow-sm text-indigo-700' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </nav>
      {tab === 'approvals' && <ApprovalQueue />}
      {tab === 'traces' && <TraceViewer />}
      {tab === 'incidents' && <Incidents />}
      {tab === 'routing' && <Routing />}
      {tab !== 'approvals' && tab !== 'traces' && tab !== 'incidents' && tab !== 'routing' && <PageNotFound />}
    </>
  )
}

export default App

// ------------------------------------------------------------------ Approvals

function ApprovalQueue() {
  const [items, setItems] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [role, setRole] = useState<'sre' | 'viewer'>('sre')
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api('/approvals/pending')
      setItems(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const decide = async (id: string, decision: 'approved' | 'rejected') => {
    try {
      await api(`/approvals/${id}`, {
        method: 'POST',
        body: JSON.stringify({ decision }),
      })
      setItems(items.filter((i: any) => i.id !== id))
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <Page title="Approval Queue">
      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="radio"
            name="role"
            checked={role === 'sre'}
            onChange={() => setRole('sre')}
            className="accent-indigo-600"
          />
          SRE (can approve)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="radio"
            name="role"
            checked={role === 'viewer'}
            onChange={() => setRole('viewer')}
            className="accent-indigo-600"
          />
          Viewer (read-only)
        </label>
        <Button onClick={load} loading={loading} className="ml-auto" variant="ghost">
          Refresh
        </Button>
      </div>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {loading ? (
        <Empty message="Loading approvals…" />
      ) : items.length === 0 ? (
        <Empty message="No pending approvals." />
      ) : (
        <div className="space-y-3">
          {items.map((item: any) => (
            <Card key={item.id} className="py-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge label={item.risk} color={item.risk === 'high' ? 'red' : item.risk === 'medium' ? 'yellow' : 'green'} />
                    <Badge label={item.agent} color="blue" />
                    <span className="text-sm text-gray-400">{new Date(item.created_at).toLocaleString()}</span>
                  </div>
                  <h3 className="font-medium text-gray-900">{item.title}</h3>
                  <p className="text-sm text-gray-500 mt-1">{item.action}</p>
                  {item.blast_radius ? (
                    <p className="text-xs text-gray-400 mt-1">Blast radius: {item.blast_radius}</p>
                  ) : null}
                  <p className="text-xs text-gray-400 mt-1">Trace: {item.trace_id}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {role === 'sre' ? (
                    <>
                      <Button
                        variant="primary"
                        className="bg-green-600 hover:bg-green-700"
                        onClick={() => decide(item.id, 'approved')}
                      >
                        Approve
                      </Button>
                      <Button
                        variant="danger"
                        onClick={() => decide(item.id, 'rejected')}
                      >
                        Reject
                      </Button>
                    </>
                  ) : (
                    <Badge label="pending" color="gray" />
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </Page>
  )
}

// ------------------------------------------------------------------ Traces

function TraceViewer() {
  const [traceId, setTraceId] = useState('')
  const [spans, setSpans] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    if (!traceId) return
    setLoading(true)
    setError('')
    try {
      const data = await api(`/traces/${encodeURIComponent(traceId)}`)
      setSpans(data.spans ?? data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Page title="Trace Viewer">
      <div className="flex gap-2 mb-4">
        <input
          value={traceId}
          onChange={(e) => setTraceId(e.target.value)}
          placeholder="paste a trace_id …"
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <Button onClick={load} disabled={!traceId || loading} variant="primary">
          {loading ? 'Loading …' : 'View trace'}
        </Button>
      </div>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {spans.length > 0 ? (
        <div className="space-y-2">
          {spans.map((span: any, idx: number) => (
            <Card key={span.span_id ?? idx} className="py-3">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <Badge label={span.kind} color="blue" />
                    <Badge label={span.status} color={span.status === 'UNSET' ? 'gray' : span.status === 'ERROR' ? 'red' : 'green'} />
                    <span className="text-xs text-gray-400 font-mono">{span.span_id}</span>
                  </div>
                  <h4 className="font-medium text-gray-900">{span.name}</h4>
                  {span.parent_span_id ? (
                    <p className="text-xs text-gray-400 mt-1">parent: {span.parent_span_id}</p>
                  ) : null}
                  {span.latency_ms != null ? (
                    <p className="text-xs text-gray-500 mt-1">latency: {span.latency_ms.toFixed(2)} ms</p>
                  ) : null}
                  {span.attributes?.llm?.cost_usd != null ? (
                    <p className="text-xs text-gray-500 mt-1">cost: ${span.attributes.llm.cost_usd.toFixed(4)}</p>
                  ) : null}
                </div>
                {span.trace_id ? (
                  <a
                    href={`https://phoenix.example.com/traces/${span.trace_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-indigo-600 hover:underline shrink-0"
                  >
                    open in Phoenix →
                  </a>
                ) : null}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Empty message="Paste a trace_id and click View trace." />
      )}
    </Page>
  )
}

// ------------------------------------------------------------------ Incidents

function Incidents() {
  const [items, setItems] = useState<any[]>([])
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

  useEffect(() => { load() }, [])

  const statusColor = (s: string) => {
    if (s === 'open') return 'red'
    if (s === 'resolved') return 'green'
    if (s === 'monitoring') return 'yellow'
    return 'gray'
  }

  return (
    <Page title="Incident Commander">
      <Button onClick={load} loading={loading} variant="ghost" className="mb-4">
        Refresh
      </Button>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {items.length === 0 ? (
        <Empty message="No incidents recorded." />
      ) : (
        <div className="space-y-3">
          {items.map((inc: any) => (
            <Card key={inc.id} className="py-4">
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
                    <p className="text-sm text-gray-400 mt-1">report pending …</p>
                  )}
                </div>
                <Badge label={inc.id} color="gray" />
              </div>
            </Card>
          ))}
        </div>
      )}
    </Page>
  )
}

// ------------------------------------------------------------------ Routing

function Routing() {
  const [state, setState] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api('/routing/state')
      setState(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <Page title="RouteBench — Routing State">
      <Button onClick={load} loading={loading} variant="ghost" className="mb-4">
        Refresh
      </Button>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {state ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <h3 className="font-medium text-gray-900 mb-3">Backend quality × cost matrix</h3>
            {state.model ? (
              <ul className="space-y-2 text-sm">
                <li className="flex justify-between"><span className="text-gray-500">Model</span><span className="font-medium">{state.model}</span></li>
                <li className="flex justify-between"><span className="text-gray-500">Provider</span><span className="font-medium">{state.provider}</span></li>
                {(Object.entries(state.quality ?? {}) as [string, number][]).map(([name, q]) => (
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
            ) : (
              <Empty message="No backend data." />
            )}
          </Card>

          <Card>
            <h3 className="font-medium text-gray-900 mb-3">Traffic weights</h3>
            {state.traffic_weights ? (
              <ul className="space-y-2 text-sm">
                {(Object.entries(state.traffic_weights) as [string, number][]).map(([model, weight]) => (
                  <li key={model} className="flex justify-between">
                    <span className="text-gray-500">{model}</span>
                    <span className="font-medium">{weight}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <Empty message="No weights recorded." />
            )}
            <div className="mt-4 pt-3 border-t border-gray-200">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Circuit breaker</span>
                <Badge label={state.circuit_open ? 'OPEN' : 'closed'} color={state.circuit_open ? 'red' : 'green'} />
              </div>
            </div>
          </Card>
        </div>
      ) : (
        <Empty message="Click Refresh to load routing state." />
      )}
    </Page>
  )
}
