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

type Approval = {
  id: string
  trace_id: string
  agent: string
  title: string
  action: string
  risk: string
  blast_radius: string
  created_at: string
}

function ApprovalQueue() {
  const [items, setItems] = useState<Approval[]>([])
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
      setItems(items.filter((i) => i.id !== id))
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
        <Button onClick={load} className="ml-auto" variant="ghost" disabled={loading}>
          Refresh
        </Button>
      </div>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {loading ? (
        <Empty message="Loading approvals..." />
      ) : items.length === 0 ? (
        <Empty message="No pending approvals." />
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
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

export default ApprovalQueue
