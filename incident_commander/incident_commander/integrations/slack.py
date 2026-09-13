"""Realistic local Slack webhook capture."""
import json
from pathlib import Path


class SlackCapture:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, channel: str, text: str, blocks: list[dict[str, object]] | None = None) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"channel": channel, "text": text, "blocks": blocks or []}) + "\n")

