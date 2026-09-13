# RouteBench decisions

- Use the shared in-process cache interface for classifier and response caching; no gateway code imports a cache vendor.
- Keep routing data-driven: backend metadata and per-task quality matrices are inputs, while policy scores quality, cost, latency, availability, SLA, context, and tenant budget.
- Represent providers through one `BackendClient` protocol; OpenAI-wire provider endpoints share the tested implementation.
- Use completed-response streaming chunks for protocol compatibility in this local control-plane build; backend-native token streaming can replace the client implementation.
- Commit the Grafana dashboard JSON without attempting to render it on this host.

