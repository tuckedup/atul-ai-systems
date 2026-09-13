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
    <div className="min-h-screen bg-gray-50">
      <PageHeader title="Trace Viewer" />
      <div className="flex gap-2 mb-4">
        <input
          value={traceId}
          onChange={(e) => setTraceId(e.target.value)}
          placeholder="paste a trace_id ..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          onClick={load}
          disabled={!traceId || loading}
          className="px-3 py-2 bg-indigo-600 text-white rounded-md text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Loading ...' : 'View trace'}
        </button>
      </div>
      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}
      {spans.length > 0 ? (
        <div className="space-y-2">
          {spans.map((span, idx) => (
            <div key={span.span_id ?? idx} className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <Badge label={span.kind} color="bg-blue-100 text-blue-800" />
                    <Badge label={span.status} color={span.status === 'UNSET' ? 'bg-gray-100 text-gray-700' : span.status === 'ERROR' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'} />
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
            </div>
          ))}
        </div>
      ) : (
        <Empty msg="Paste a trace_id and click View trace." />
      )}
    </div>
  )
}

export default TraceViewer
