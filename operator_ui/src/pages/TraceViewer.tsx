import { useState } from 'react'

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

type Span = {
  span_id: string
  parent_span_id: string | null
  name: string
  kind: string
  status: string
  latency_ms: number
  attributes: Record<string, unknown>
  trace_id?: string
}

function TraceViewer() {
  const [traceId, setTraceId] = useState('')
  const [spans, setSpans] = useState<Span[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    if (!traceId) return
    setLoading(true)
    setError('')
    try {
      const data = await api(`/traces/${encodeURIComponent(traceId)}`)
      const traceData = data.spans ?? data
      setSpans(Array.isArray(traceData) ? traceData : [])
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
          placeholder="paste a trace_id ..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <Button onClick={load} disabled={!traceId || loading} variant="primary">
          {loading ? 'Loading ...' : 'View trace'}
        </Button>
      </div>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {spans.length > 0 ? (
        <div className="space-y-2">
          {spans.map((span, idx) => (
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
                  {span.attributes && 'llm' in span.attributes && 'cost_usd' in (span.attributes as any).llm && (
                    <p className="text-xs text-gray-500 mt-1">cost: ${((span.attributes as any).llm.cost_usd as number).toFixed(4)}</p>
                  )}
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

export default TraceViewer
