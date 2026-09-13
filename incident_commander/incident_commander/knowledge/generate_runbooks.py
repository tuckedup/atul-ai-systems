from pathlib import Path


def main() -> None:
    target = Path(__file__).parent / "runbooks"
    target.mkdir(parents=True, exist_ok=True)
    for number in range(20):
        roles = "[sre]" if number % 2 else "[sre, viewer]"
        service = "database" if number % 3 == 0 else "api"
        text = (
            f"---\ntitle: {service.title()} recovery {number + 1}\nroles: {roles}\n---\n"
            f"Validate {service} health, compare the recent deployment, and collect evidence. "
            "Any restart, scale, rollback, database modification, or merge requires approval.\n"
        )
        (target / f"runbook-{number + 1:02d}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

