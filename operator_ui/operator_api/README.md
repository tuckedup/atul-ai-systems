# Operator API

Local FastAPI fixture service for the Operator UI. It exposes health, approval,
trace, incident, and routing endpoints without Docker or external providers.

Run `python seed.py` to reset deterministic fixtures, then start
`uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777` from `operator_ui/`.
