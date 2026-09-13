# ForgeCode decisions

- Use the amended subprocess sandbox with a unique fixture copy, hard timeout, recursive RSS cap, and explicit repository-root rejection.
- Initialize a private git repository inside each copied fixture so diffs never resolve against the portfolio repository.
- Select model tiers by task class, retry count, and observed failure; concrete model names remain environment-configurable.
- Use the shared database DSN to select SQLite or PostgreSQL checkpoint adapters without changing graph callers.
- Keep the independent reviewer boundary as a pure `reviewer_messages` function so tests can prove private plan/context exclusion.
- Accept standard `OPENAI_BASE_URL`/`OPENAI_API_KEY` names as aliases for the `AISYS_`-prefixed settings so the shared client works with provider tooling without copying secrets.
