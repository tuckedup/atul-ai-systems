from dataclasses import dataclass


@dataclass(frozen=True)
class JiraTicket:
    key: str
    summary: str
    status: str


FIXTURES = [JiraTicket("OPS-417", "Checkout errors after release", "Open")]

