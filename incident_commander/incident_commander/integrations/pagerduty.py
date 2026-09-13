from dataclasses import dataclass


@dataclass(frozen=True)
class PagerDutyAlert:
    id: str
    service: str
    severity: str
    summary: str


FIXTURES = [PagerDutyAlert("PD-1001", "checkout-api", "critical", "Error rate exceeded 15%")]

