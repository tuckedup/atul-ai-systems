# Incident Commander decisions

- Enforce RBAC before retrieval ranking so unauthorized runbooks never enter model context.
- Redact emails, bearer credentials, API keys, tokens, and passwords before any model boundary.
- Use the single database DSN for approval, audit, per-incident memory, and cross-incident memory.
- Use real GitHub OAuth and issue APIs; credentials and an interactive authorization grant are required for live proof.
- Use realistic local Slack, PagerDuty, and Jira fixtures; Prometheus and Kubernetes live proofs are Docker-blocked.
- In degraded M3 mode, use LangGraph's SQLite saver with synchronous durability and cross the approved execution boundary in a child Python process; do not claim that local subprocess proof as Kubernetes execution.
