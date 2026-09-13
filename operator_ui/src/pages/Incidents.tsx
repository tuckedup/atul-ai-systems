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

  useEffect(() => { load() }, [])

  const statusColor = (s: string) => {
    if (s === 'open') return 'red'
    if (s === 'resolved') return 'green'
    if (s === 'monitoring') return 'yellow'
    return 'gray'
  }

  return (
    <Page title="Incident Commander">
      <Button onClick={load} disabled={loading} variant="ghost" className="mb-4">
        Refresh
      </Button>

      {error ? (
        <div className="bg-red-50 text-red-700 px-4 py-3 rounded border border-red-200 mb-4">{error}</div>
      ) : null}

      {items.length === 0 ? (
        <Empty message="No incidents recorded." />
      ) : (
        <div className="space-y-3">
          {items.map((inc) => (
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
                    <p className="text-sm text-gray-400 mt-1">report pending ...</p>
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

export default Incidents
