# Project blockers

## Incident Commander M5

**BLOCKED: awaiting human OAuth setup**

The enterprise GitHub integration requires an interactive GitHub OAuth grant that a human must authorize. No client authorization, access token, or external GitHub write was attempted. Once OAuth is configured, M5 can validate the authenticated enterprise workflow end to end.

## Resolved blockers

- Docker infrastructure is available and the shared stack runs locally.
- RouteBench's GPU experiments completed with the documented vLLM compatibility fallback.
- Incident Commander M6 live inference is complete: 15/15 calls succeeded across five scenarios and three seeds.
