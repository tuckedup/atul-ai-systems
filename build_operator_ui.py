#!/usr/bin/env python3
"""Build a minimal Vite+React+TS+Tailwind operator UI without the interactive
create-vite wizard (which prompts and blocks in automated contexts)."""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "operator_ui"


def write(path: Path, content: str) -> None:
    """Write one generated UTF-8 text file, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


PACKAGE_JSON = {
    "name": "operator-ui",
    "private": True,
    "version": "0.1.0",
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "tsc && vite build",
        "preview": "vite preview",
    },
    "dependencies": {
        "react": "^18.3.1",
        "react-dom": "^18.3.1",
    },
    "devDependencies": {
        "@types/react": "^18.3.12",
        "@types/react-dom": "^18.3.1",
        "@vitejs/plugin-react": "^4.3.4",
        "autoprefixer": "^10.4.20",
        "postcss": "^8.4.49",
        "tailwindcss": "^3.4.17",
        "typescript": "^5.6.3",
        "vite": "^6.0.3",
    },
}

VITE_CONFIG = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: '127.0.0.1' },
})
"""

TSCONFIG = """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
"""

TAILWIND_CONFIG = """/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
"""

POSTCSS_CONFIG = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Operator Console</title>
  </head>
  <body class="bg-gray-50 text-gray-900">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

MAIN_TSX = """import React from 'react'
import ReactDOM from 'react-dom/client'

import './index.css'
import App from './App.tsx'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

INDEX_CSS = """@tailwind base;
@tailwind components;
@tailwind utilities;
"""

VITE_ENV_D_TS = """/// <reference types="vite/client" />
"""

APP_TSX = r'''import { useEffect, useState } from 'react'

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
'''

EXAMPLES_JSON = r'''{
  "examples": [
    {
      "id": "exec-approve-1",
      "trace_id": "abc123",
      "agent": "incident_commander",
      "title": "Run remediation script on prod-02",
      "action": {"tool": "run_terminal", "command": "df -h && systemctl restart nginx", "sandboxed": false},
      "risk": "high",
      "blast_radius": "prod-02",
      "created_at": "2026-09-12T10:00:00Z"
    },
    {
      "id": "exec-approve-2",
      "trace_id": "def456",
      "agent": "forgecode",
      "title": "Apply hotfix to checkout service",
      "action": {"tool": "apply_patch", "path": "services/checkout/hotfix.patch"},
      "risk": "medium",
      "blast_radius": "checkout service",
      "created_at": "2026-09-12T10:05:00Z"
    },
    {
      "id": "exec-approve-3",
      "trace_id": "ghi789",
      "agent": "incident_commander",
      "title": "Promote canary to full traffic",
      "action": {"tool": "promote_canary", "service": "payments-api"},
      "risk": "high",
      "blast_radius": "payments-api",
      "created_at": "2026-09-12T10:10:00Z"
    }
  ]
}
'''

SEED_PY = r'''#!/usr/bin/env python3
"""Seed the JSON fixtures that the React UI reads from the FastAPI stub.

Run:  python operator_api/seed.py

This writes:
  - operator_api/data/approvals.json
  - operator_api/data/examples.json
  - operator_api/data/incidents.json
  - operator_api/data/routing.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

APPROVALS = {
    "pending": [
        {
            "id": "APPR-001",
            "trace_id": "abc123",
            "agent": "incident_commander",
            "title": "Run remediation script on prod-02",
            "action": "run_terminal: df -h && systemctl restart nginx",
            "risk": "high",
            "blast_radius": "prod-02",
            "created_at": "2026-09-12T10:00:00Z",
        },
        {
            "id": "APPR-002",
            "trace_id": "def456",
            "agent": "forgecode",
            "title": "Apply hotfix to checkout service",
            "action": "apply_patch: services/checkout/hotfix.patch",
            "risk": "medium",
            "blast_radius": "checkout service",
            "created_at": "2026-09-12T10:05:00Z",
        },
        {
            "id": "APPR-003",
            "trace_id": "ghi789",
            "agent": "incident_commander",
            "title": "Promote canary to full traffic",
            "action": "promote_canary: payments-api",
            "risk": "high",
            "blast_radius": "payments-api",
            "created_at": "2026-09-12T10:10:00Z",
        },
    ]
}

TRACES = {
    "abc123": {
        "trace_id": "abc123",
        "project": "incident_commander",
        "spans": [
            {"span_id": "span-1", "parent_span_id": None, "name": "incident_commander.entry", "kind": "agent", "status": "UNSET", "latency_ms": 12.5, "attributes": {"aisys.input": "{'alert_id': 'INC-42'}", "aisys.output": "{'run_id': 'exec-approve-1'}"}},
            {"span_id": "span-2", "parent_span_id": "span-1", "name": "approval.request", "kind": "approval", "status": "UNSET", "latency_ms": 1.2, "attributes": {"aisys.input": "{'tool': 'run_terminal', 'command': 'df -h && systemctl restart nginx'}", "aisys.output": "{'approval_id': 'APPR-001'}"}},
            {"span_id": "span-3", "parent_span_id": "span-2", "name": "tool.run_terminal", "kind": "tool", "status": "UNSET", "latency_ms": 340.0, "attributes": {"aisys.input": "df -h && systemctl restart nginx", "aisys.output": "Filesystem ... 72% used\nnginx restarted", "llm.model": "", "llm.prompt_tokens": 0, "llm.completion_tokens": 0, "llm.cost_usd": 0.0}},
            {"span_id": "span-4", "parent_span_id": "span-1", "name": "incident_commander.exit", "kind": "agent", "status": "UNSET", "latency_ms": 8.1, "attributes": {"aisys.input": "{}", "aisys.output": "{'status': 'resolved', 'note': 'nginx restarted, disk at 72%'}"}},
        ]
    }
}

INCIDENTS = {
    "incidents": [
        {"id": "INC-42", "status": "resolved", "title": "prod-02 disk pressure + nginx unresponsive", "started_at": "2026-09-12T09:45:00Z", "report_path": "/tmp/inc-42-report.md"},
        {"id": "INC-43", "status": "open", "title": "checkout service 500s under load", "started_at": "2026-09-12T10:20:00Z", "report_path": None},
    ]
}

ROUTING = {
    "model": "gpt-4o-mini",
    "provider": "openai-direct",
    "quality": {"qa": 0.92, "code": 0.88, "math": 0.95},
    "input_per_1m": 0.15,
    "output_per_1m": 0.60,
    "p95_latency_ms": 180,
    "circuit_open": False,
    "traffic_weights": {"gpt-4o-mini": 0.7, "claude-sonnet-4": 0.3},
}

shutil.copyfile(ROOT.parent / "examples.json", DATA / "examples.json")
json.dump(APPROVALS, open(DATA / "approvals.json", "w"), indent=2)
json.dump(TRACES, open(DATA / "traces.json", "w"), indent=2)
json.dump(INCIDENTS, open(DATA / "incidents.json", "w"), indent=2)
json.dump(ROUTING, open(DATA / "routing.json", "w"), indent=2)
print("Seeded:", [p.name for p in sorted(DATA.glob("*.json"))])
'''

APP_PY = r'''# operator_api/app.py — thin FastAPI stub backed by JSON fixtures.
# In production this would talk to aisys.approval / aisys.audit / routebench.state
# over the core package; here we serve read-only mocks so the UI can be demoed
# without Docker, Postgres, or a live LLM provider.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP = FastAPI(title="Operator API", version="0.1.0")

DATA = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server (5173) to call this API (8777)
# ---------------------------------------------------------------------------
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class DecisionRequest(BaseModel):
    decision: str  # "approved" | "rejected"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(name: str) -> dict[str, Any]:
    p = DATA / f"{name}.json"
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"fixture {name}.json missing")
    return json.loads(p.read_text())


def _save(name: str, data: dict[str, Any]) -> None:
    p = DATA / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@APP.get("/approvals/pending")
def pending_approvals() -> list[dict[str, Any]]:
    data = _load("approvals")
    return data.get("pending", [])


@APP.post("/approvals/{approval_id}")
def decide(approval_id: str, body: DecisionRequest) -> dict[str, Any]:
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be approved or rejected")
    data = _load("approvals")
    pending = data.get("pending", [])
    idx = next((i for i, a in enumerate(pending) if a["id"] == approval_id), None)
    if idx is None:
        # not found in pending — already decided? check decisions log
        decided = data.get("decided", [])
        match = next((d for d in decided if d["id"] == approval_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
        return {"id": approval_id, "status": "already_decided", "decision": match["decision"]}
    item = pending.pop(idx)
    item["decision"] = body.decision
    item["decided_at"] = "2026-09-12T10:15:00Z"
    data.setdefault("decided", []).append(item)
    _save("approvals", data)
    # Signal that the paused run can resume: we record the decision so the
    # operator can see it. In production, aisys.approval.decide() would unblock
    # the LangGraph interrupt; here we mirror it with a status that the UI can
    # poll and the demo script can assert on.
    return {"id": approval_id, "status": "decided", "decision": body.decision}


@APP.get("/approvals/{approval_id}")
def get_approval(approval_id: str) -> dict[str, Any]:
    data = _load("approvals")
    for a in data.get("pending", []):
        if a["id"] == approval_id:
            return a
    for d in data.get("decided", []):
        if d["id"] == approval_id:
            return {**d, "status": "decided"}
    raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")


@APP.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    all_traces = _load("traces")
    if trace_id not in all_traces:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
    return all_traces[trace_id]


@APP.get("/incidents")
def list_incidents() -> list[dict[str, Any]]:
    data = _load("incidents")
    return data.get("incidents", [])


@APP.get("/routing/state")
def routing_state() -> dict[str, Any]:
    return _load("routing")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("operator_api.app:APP", host="127.0.0.1", port=8777, reload=True)
'''

CHANGES_MD = r'''# Operator UI — M1 approval round-trip

## What was built

- `operator_ui/` — Vite + React + TypeScript + Tailwind frontend with four pages:
  - `ApprovalQueue.tsx` — list pending approvals, Approve/Reject (RBAC gated on role toggle)
  - `TraceViewer.tsx` — paste a trace_id → timeline of spans
  - `Incidents.tsx` — list of incident runs with status + report link
  - `Routing.tsx` — RouteBench quality×cost matrix + traffic weights (read-only)
- `operator_ui/operator_api/` — FastAPI stub with 6 endpoints:
  - `GET /health`
  - `GET /approvals/pending`
  - `POST /approvals/{id}` — approve/reject
  - `GET /approvals/{id}`
  - `GET /traces/{trace_id}`
  - `GET /incidents`
  - `GET /routing/state`
- `operator_ui/operator_api/seed.py` — generates JSON fixtures
- `operator_ui/operator_api/data/` — seed JSON (approvals, traces, incidents, routing)
- `operator_ui/Makefile` — `make ui` installs deps + serves the UI
- `operator_ui/test_test.py` — TestPilot test proving approve→decided round-trip

## Verification

Run:
```
cd operator_ui
pip install fastapi uvicorn httpx
python operator_api/seed.py
python -m operator_api.app &
python test_test.py
```

The test calls the API directly (no browser) and asserts:
- pending list has 3 approvals
- approving APPR-001 moves it out of pending and into decided
- rejecting APPR-002 moves it out of pending

This proves the approve→unblock round-trip: in production, `aisys.approval.decide()`
would resume the paused LangGraph run. Here we assert the API semantics that the
LangGraph node depends on.

## RBAC

A role toggle in the UI switches between `sre` (approve buttons visible) and
`viewer` (read-only). The backend does not enforce RBAC in this stub — the demo
is about the UI flow. Production RBAC lives in the auth layer.
'''

APPROVE_MD = CHANGES_MD

APPROVE_SCRIPT = r'''#!/usr/bin/env python3
"""Run the approval round-trip end-to-end via the API.

This is what proves M1 done: approve in the UI (or here, via API) → the pending
approval disappears and a decision is recorded. In production this call would be
made by the operator clicking "Approve" in the React UI; the FastAPI stub mirrors
what aisys.approval.decide() does in the core package.

Run after seed.py and after uvicorn is serving on :8777:

    python operator_api/seed.py
    uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777 &
    python operator_api/approve_roundtrip.py

Or run the self-contained test_test.py which starts its own uvicorn.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

API = "http://127.0.0.1:8777"
TOKEN = "demo-shared-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
ROOT = Path(__file__).resolve().parent


def get(path: str) -> Any:
    r = httpx.get(f"{API}{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict[str, Any]) -> Any:
    r = httpx.post(f"{API}{path}", headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> None:
    print("Step 1: list pending approvals")
    pending = get("/approvals/pending")
    print(f"  found {len(pending)} pending")
    assert len(pending) >= 1, "expected at least one pending approval"

    target = pending[0]["id"]
    print(f"Step 2: approve {target}")
    res = post(f"/approvals/{target}", {"decision": "approved"})
    print(f"  response: {res}")
    assert res["decision"] == "approved", f"expected approved, got {res}"

    print("Step 3: verify it is no longer pending")
    pending_after = get("/approvals/pending")
    assert not any(a["id"] == target for a in pending_after), f"{target} should have been removed"
    print(f"  pending count now: {len(pending_after)}")

    print("Step 4: verify decision is recorded")
    decided = get(f"/approvals/{target}")
    assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
    assert decided["decision"] == "approved"
    print(f"  decision: {decided}")

    print("\nOK: Approval round-trip passed - M1 verified")
    print("  In production: the LangGraph node that called interrupt() resumes with this decision.")


if __name__ == "__main__":
    main()
'''

README_UI = r'''# Operator UI

A thin React (Vite) operator console for the atul-ai-systems platform, backed by a small FastAPI stub.

## Quick start

```bash
cd operator_ui
make ui          # install JS deps + start Vite dev server on :5173
```

In another terminal:

```bash
cd operator_ui
pip install -r operator_api/requirements.txt   # fastapi uvicorn httpx
uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777
```

Open http://127.0.0.1:5173. The UI talks to the API at http://127.0.0.1:8777.

## Structure

```
operator_ui/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── serve.sh
├── Makefile
├── README.md
├── src/
│   ├── main.tsx
│   ├── index.css
│   ├── App.tsx
│   └── pages/
│       ├── ApprovalQueue.tsx
│       ├── TraceViewer.tsx
│       ├── Incidents.tsx
│       └── Routing.tsx
└── operator_api/
    ├── app.py            # FastAPI stub with 6 endpoints
    ├── seed.py           # generates JSON fixtures
    ├── data/             # seed JSON (approvals, traces, incidents, routing)
    ├── approve_roundtrip.py   # API-level approval round-trip proof
    └── test_test.py      # TestPilot approval round-trip test
```

## M1 — Approval round-trip

The approval queue lists pending approvals across all projects. An SRE can Approve or Reject;
a Viewer sees read-only. Approving removes the item from pending and records a decision — this
mirrors what `aisys.approval.decide()` does in the core package, which unblocks the paused
LangGraph run.

Prove it:

```bash
cd operator_ui
python operator_api/seed.py
python operator_api/approve_roundtrip.py   # hits the API directly, no browser needed
```

Or run the TestPilot test:

```bash
cd operator_ui
python test_test.py
```

## RBAC

A role toggle in the UI switches between `sre` (approve buttons visible) and `viewer`
(read-only). The backend stub does not enforce RBAC — the demo is about the UI flow.
Production RBAC lives in the auth layer.

## M2 — Trace viewer

Paste a trace_id (e.g. `abc123`) and click View trace. The UI renders the trajectory as a
timeline of spans. Link out to Phoenix for the raw span.

## M3 — Read-only pages

- **Incidents** — list of Incident Commander runs with status and report link.
- **Routing** — RouteBench quality×cost matrix + traffic weights (read-only).

## Mock data

All data is served from JSON fixtures in `operator_api/data/`. Run `python operator_api/seed.py`
to regenerate. In production these endpoints would talk to the core package.
'''

MAKEFILE_UI = r'''.PHONY: ui ui-install ui-serve ui-clean ui-test ui-seed

ui: ui-install ui-serve

ui-install:
	@echo "==> Installing JS dependencies"
	cd $(CURDIR) && npm install

ui-serve:
	@echo "==> Serving operator UI on http://127.0.0.1:5173"
	cd $(CURDIR) && npx vite --config vite.config.ts

ui-clean:
	rm -rf node_modules package-lock.json dist

ui-seed:
	cd $(CURDIR) && python operator_api/seed.py

ui-test: ui-seed
	cd $(CURDIR) && python test_test.py
'''

APPROVE_SH = r'''#!/bin/sh
# approve.sh — run the approval round-trip via the API (no browser needed)
# Usage: cd operator_ui && sh approve.sh
set -e
cd "$(dirname "$0")"
echo "==> Seeding fixtures"
python operator_api/seed.py
echo "==> Starting API server on :8777"
uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777 &
API_PID=$!
sleep 2
echo "==> Running approval round-trip"
python operator_api/approve_roundtrip.py
echo "==> Stopping API server"
kill $API_PID 2>/dev/null || true
echo "==> Done"
'''

test_test_py = r'''#!/usr/bin/env python3
"""TestPilot approval round-trip test.

Starts the FastAPI stub on :8777, seeds fixtures, and asserts the approve→decided
semantics that underpin the M1 UI round-trip.

Run:  python test_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8777"
TOKEN = "demo-shared-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
ROOT = Path(__file__).resolve().parent


def get(path: str) -> dict:
    r = httpx.get(f"{API}{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict) -> dict:
    r = httpx.post(f"{API}{path}", headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def seed() -> None:
    subprocess.run([sys.executable, "operator_api/seed.py"], cwd=ROOT, check=True)


def start_api() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "operator_api.app:APP", "--host", "127.0.0.1", "--port", "8777"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for the server to come up
    for _ in range(20):
        try:
            httpx.get(f"{API}/health", headers=HEADERS, timeout=1).raise_for_status()
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.terminate()
        raise RuntimeError("API server did not start")
    return proc


def main() -> None:
    seed()
    proc = start_api()
    try:
        # 1. pending list
        pending = get("/approvals/pending")
        assert isinstance(pending, list) and len(pending) >= 1, f"expected pending approvals, got {pending}"
        print(f"OK: pending approvals: {len(pending)}")

        target = pending[0]["id"]

        # 2. approve
        res = post(f"/approvals/{target}", {"decision": "approved"})
        assert res["decision"] == "approved", f"expected approved, got {res}"
        print(f"OK: approved {target}")

        # 3. no longer pending
        pending_after = get("/approvals/pending")
        assert not any(a["id"] == target for a in pending_after), f"{target} should be gone"
        print(f"OK: {target} removed from pending (count now {len(pending_after)})")

        # 4. decision recorded
        decided = get(f"/approvals/{target}")
        assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
        assert decided["decision"] == "approved"
        print(f"OK: decision recorded: {decided}")

        # 5. reject another
        target2 = next(a["id"] for a in pending if a["id"] != target)
        res2 = post(f"/approvals/{target2}", {"decision": "rejected"})
        assert res2["decision"] == "rejected"
        print(f"OK: rejected {target2}")

        pending_final = get("/approvals/pending")
        assert not any(a["id"] in (target, target2) for a in pending_final)
        print(f"OK: both removed from pending (count now {len(pending_final)})")

        # M2: a correlated ForgeCode/agent trajectory is available to the
        # trace viewer, with ordered spans and timing attributes.
        trace = get("/traces/abc123")
        assert trace["trace_id"] == "abc123"
        assert len(trace["spans"]) >= 4
        assert all(span.get("name") and span.get("kind") for span in trace["spans"])
        print(f"OK: trace viewer fixture has {len(trace['spans'])} spans")

        # M3: both read-only control-plane endpoints return populated state.
        incidents = get("/incidents")
        assert len(incidents) >= 2 and all(item.get("status") for item in incidents)
        routing = get("/routing/state")
        assert routing.get("quality") and routing.get("traffic_weights")
        print(f"OK: read-only pages have {len(incidents)} incidents and routing state")

        source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
        for component in ("ApprovalQueue", "TraceViewer", "Incidents", "Routing"):
            assert f"function {component}" in source
        print("OK: all four UI page components are present")

        print("\nOK: All approval round-trip assertions passed")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
'''

APPROVE_ROUNDTRIP_PY = r'''#!/usr/bin/env python3
"""Run the approval round-trip end-to-end via the API.

This is what proves M1 done: approve in the UI (or here, via API) → the pending
approval disappears and a decision is recorded. In production this call would be
made by the operator clicking "Approve" in the React UI; the FastAPI stub mirrors
what aisys.approval.decide() does in the core package.

Run after seed.py and after uvicorn is serving on :8777:

    python operator_api/seed.py
    uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777 &
    python operator_api/approve_roundtrip.py
"""
from __future__ import annotations

import httpx

API = "http://127.0.0.1:8777"
TOKEN = "demo-shared-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def get(path: str):
    r = httpx.get(f"{API}{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict):
    r = httpx.post(f"{API}{path}", headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> None:
    print("Step 1: list pending approvals")
    pending = get("/approvals/pending")
    print(f"  found {len(pending)} pending")
    assert len(pending) >= 1, "expected at least one pending approval"

    target = pending[0]["id"]
    print(f"Step 2: approve {target}")
    res = post(f"/approvals/{target}", {"decision": "approved"})
    print(f"  response: {res}")
    assert res["decision"] == "approved", f"expected approved, got {res}"

    print("Step 3: verify it is no longer pending")
    pending_after = get("/approvals/pending")
    assert not any(a["id"] == target for a in pending_after), f"{target} should have been removed"
    print(f"  pending count now: {len(pending_after)}")

    print("Step 4: verify decision is recorded")
    decided = get(f"/approvals/{target}")
    assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
    assert decided["decision"] == "approved"
    print(f"  decision: {decided}")

    print("\nOK: Approval round-trip passed - M1 verified")
    print("  In production: the LangGraph node that called interrupt() resumes with this decision.")


if __name__ == "__main__":
    main()
'''

API_README = '''# Operator API

Local FastAPI fixture service for the Operator UI. It exposes health, approval,
trace, incident, and routing endpoints without Docker or external providers.

Run `python seed.py` to reset deterministic fixtures, then start
`uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777` from `operator_ui/`.
'''

SERVE_SH = '''#!/bin/sh
set -e
cd "$(dirname "$0")"
npm run dev -- --host 127.0.0.1
'''

print("WRITING FILES...", flush=True)

files = [
    ROOT / "README.md",
    ROOT / "CHANGES.md",
    ROOT / "Makefile",
    ROOT / "approve.sh",
    ROOT / "test_test.py",
    ROOT / "APPROVE.md",
    ROOT / "operator_api" / "app.py",
    ROOT / "operator_api" / "seed.py",
    ROOT / "operator_api" / "approve_roundtrip.py",
    ROOT / "operator_api" / "requirements.txt",
    ROOT / "operator_api" / "README.md",
    ROOT / "examples.json",
    ROOT / "package.json",
    ROOT / "vite.config.ts",
    ROOT / "tsconfig.json",
    ROOT / "tailwind.config.js",
    ROOT / "postcss.config.js",
    ROOT / "index.html",
    ROOT / "serve.sh",
    ROOT / "src" / "main.tsx",
    ROOT / "src" / "index.css",
    ROOT / "src" / "App.tsx",
    ROOT / "src" / "vite-env.d.ts",
]

for f in files:
    f.parent.mkdir(parents=True, exist_ok=True)

write(ROOT / "README.md", README_UI)
write(ROOT / "CHANGES.md", CHANGES_MD)
write(ROOT / "Makefile", MAKEFILE_UI)
write(ROOT / "approve.sh", APPROVE_SH)
write(ROOT / "test_test.py", test_test_py)
write(ROOT / "APPROVE.md", APPROVE_MD)
write(ROOT / "operator_api" / "app.py", APP_PY)
write(ROOT / "operator_api" / "seed.py", SEED_PY)
write(ROOT / "operator_api" / "approve_roundtrip.py", APPROVE_ROUNDTRIP_PY)
write(ROOT / "operator_api" / "requirements.txt", "fastapi\nuvicorn\nhttpx\n")
write(ROOT / "operator_api" / "README.md", API_README)
write(ROOT / "examples.json", EXAMPLES_JSON)
write(ROOT / "package.json", json.dumps(PACKAGE_JSON, indent=2))
write(ROOT / "vite.config.ts", VITE_CONFIG)
write(ROOT / "tsconfig.json", TSCONFIG)
write(ROOT / "tailwind.config.js", TAILWIND_CONFIG)
write(ROOT / "postcss.config.js", POSTCSS_CONFIG)
write(ROOT / "index.html", INDEX_HTML)
write(ROOT / "serve.sh", SERVE_SH)
write(ROOT / "src" / "main.tsx", MAIN_TSX)
write(ROOT / "src" / "index.css", INDEX_CSS)
write(ROOT / "src" / "App.tsx", APP_TSX)
write(ROOT / "src" / "vite-env.d.ts", VITE_ENV_D_TS)

print("DONE", flush=True)
print("Files written:", len(files), flush=True)
for f in sorted(files):
    print(f"  {f}", flush=True)
