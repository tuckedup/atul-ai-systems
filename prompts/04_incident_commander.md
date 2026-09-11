# PROMPT 04 — AI Incident Commander: enterprise agentic workflow (Phase 3)

## Goal
Prove the enterprise loop end to end: alert → multi-agent investigation → remediation proposal → human
approval → execute → post-incident report, with durable pause/resume, per-agent context scoping,
short+long-term memory, RBAC on retrieval, one real OAuth'd integration, and a verifiable audit trail.
All model calls go through RouteBench. All cross-cutting behavior comes from `aisys.*`.

## Agents (LangGraph nodes, scoped context)
| Node | Gets only | Produces |
|---|---|---|
| `coordinator` | alert, service catalog | investigation plan, fan-out |
| `logs_agent` | log slice ± 15 min, error signatures | anomaly summary |
| `dependency_agent` | service graph, recent deploys, k8s state | suspect components |
| `code_agent` | stack trace, relevant files, recent commits (via ForgeCode context engine) | candidate root cause + diff |
| `root_cause_agent` | the three summaries | ranked hypotheses with evidence |
| `remediation_agent` | top hypothesis, runbooks (RBAC-filtered) | proposed actions with blast radius |
| `risk_reviewer` | proposal, change policy, similar past incidents (long-term memory) | risk score, approval requirement |

## Folder layout to create
```
incident_commander/
├── agents/                 # one file per node above + graph.py (StateGraph, Postgres checkpointer)
├── integrations/
│   ├── github_oauth.py     # REAL: OAuth app, creates issue + PR comment on the incident repo
│   ├── slack.py            # may be mocked (fixture webhook capture) OR real — pick one and record in DECISIONS.md
│   ├── prometheus.py       # real, against infra prometheus with a synthetic-load demo service
│   ├── kubernetes.py       # kind cluster; scale/rollback/restart are real kubectl calls behind approval
│   ├── pagerduty.py jira.py  # mocked with realistic fixtures
│   └── mcp_server.py       # exposes all of the above as MCP tools via aisys.tools.serve_mcp
├── memory/
│   ├── short_term.py       # per-incident working memory: evidence, hypotheses, decisions (checkpointed)
│   └── long_term.py        # cross-incident store: embeddings of past incidents + outcomes; recall by similarity
├── knowledge/
│   ├── runbooks/           # ≥20 runbooks with `roles: [sre]` or `roles: [sre, viewer]` frontmatter
│   └── retrieval.py        # RBAC-filtered retrieval: same query, role → different results (test proves it)
├── demo/
│   ├── service/            # tiny FastAPI app + chaos flag that raises error rate 1% → 18%
│   └── scenarios/          # ≥5 incidents: bad deploy, dep outage, memory leak, config drift, cert expiry
├── reports/                # generated post-incident reports (markdown), one per run
├── tests/  Makefile  STATUS.md  DECISIONS.md  README.md
```

## Milestones
**M1 — demo service + real signals.** `make chaos` raises error rate; Prometheus alert fires; alert reaches
`coordinator`. Verify: alert payload appears as first audit row of a new incident.

**M2 — graph + scoping.** All seven nodes; a test asserts each node's prompt contains only its allowed
context (e.g., `logs_agent` never sees source code). Verify: `tests/test_context_scoping.py`.

**M3 — approval + durability.** Read-only actions auto; `restart`, `scale`, `rollback`, `db_modify`,
`merge_pr` require approval. Kill the process while waiting for approval; restart; `ic approve <id>` resumes
and executes the kubectl action. Verify: `tests/test_pause_resume.py`.

**M4 — memory.** Run the same scenario twice; second run's `risk_reviewer` cites the first incident from
long-term memory and its time-to-root-cause is lower. Verify: test asserts recall + report cites prior id.

**M5 — enterprise surface.** RBAC retrieval test (two roles, different runbooks). GitHub OAuth flow works
and the agent files an issue with the report. `audit verify` passes; tamper test fails as expected.
PII: log lines are redacted (emails, tokens) before reaching any model — test with seeded PII.

**M6 — evals + report.** `evals/` with ≥5 scenarios × 3 seeds: root-cause accuracy, time-to-proposal,
actions-requiring-approval count, unsafe-action-attempts (must be 0), tokens/cost per incident. Report
generator writes `reports/<incident_id>.md` (timeline, evidence, decision, approver, outcome).

## Also write (this is the customer-facing artifact gap)
- `docs/sow/incident_commander.md`: one page written as if a customer said "our on-call is drowning, can
  AI help?" — scope, out of scope, success metrics, phases, risks, what we need from them.
- `docs/adr/incident_commander.md`: one page — why LangGraph + Postgres checkpointer, why RBAC at retrieval
  not at prompt, why hash-chained audit, alternatives rejected.

## DEFINITION OF DONE
- `make demo` runs a full incident to an approved rollback and a filed GitHub issue.
- Pause/resume, RBAC, audit-verify, PII-redaction, memory-recall tests all green.
- `make bench` prints the scenario table; README two ways with the numbers.
- STATUS.md M1–M6 checked.
