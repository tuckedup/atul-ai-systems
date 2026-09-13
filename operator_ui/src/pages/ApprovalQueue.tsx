import { useState } from 'react'

const API = 'http://127.0.0.1:8777'
const TOKEN = 'demo-shared-token'

const ROLE: { sre: boolean; viewer: boolean } = { sre: true, viewer: false }

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
  const [error, setError] = useState('')
  const [role, setRole] = useState<boolean>(ROLE.sre)

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
    <div className="min-h-screen bg-gray-50">
      <PageHeader title="Approval Queue" />
      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={role}
            onChange={(e) => setRole(e.target.checked)}
            className="rounded"
          />
          SRE (can approve)
        </label>
        <Button onClick={load} disabled={loading} variant="ghost">
          Refresh
        </Button>
      </div>
      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}
      {loading ? (
        <Empty msg="Loading approvals..." />
      ) : items.length === 0 ? (
        <Empty msg="No pending approvals." />
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <div key={item.id} className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge
                      label={item.risk}
                      color={item.risk === 'high' ? 'bg-red-100 text-red-800' : item.risk === 'medium' ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}
                    />
                    <Badge label={item.agent} color="bg-blue-100 text-blue-800" />
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
                  {role ? (
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
                    <Badge label="pending" color="bg-gray-100 text-gray-700" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ApprovalQueue
