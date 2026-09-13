# Blocked verification

- Core M1 infrastructure boot / Phoenix UI proof — BLOCKED: requires Docker
- Core definition-of-done Phoenix trace visibility proof — BLOCKED: requires Docker
- RouteBench M2 self-hosted vLLM backend proof — BLOCKED: requires Docker
- RouteBench M3 live vLLM inference experiment matrix — BLOCKED: requires Docker
- ForgeCode M5 official 20-task SWE-bench Verified run — BLOCKED: the official harness requires a container runtime; Docker is not installed
- ForgeCode M6 ten-organic-trajectory evidence — BLOCKED on source data: `.local/traces.jsonl` contains exactly 1 genuine failed correlated trajectory after excluding 400 expected `GraphInterrupt` spans; no synthetic failures were substituted, so 9 additional genuine failures are still needed
- ForgeCode M6 hosted failing-run URL — BLOCKED: no hosted CI run exists in this local checkout and no remote workflow dispatch or push was authorized
- Incident Commander M5 live GitHub issue proof — BLOCKED: requires an interactive OAuth grant and authorization for the external write
- Incident Commander M6 live accuracy, time-to-root-cause, and cost measurements — NOT MEASURED; the earlier provider-quota blocker has cleared, but that benchmark was outside this ForgeCode task
