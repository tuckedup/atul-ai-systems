"""Local report and memory demonstration without external infrastructure."""
from aisys.settings import settings

from .memory.long_term import LongTermMemory
from .reports import write_report


def main() -> None:
    memory = LongTermMemory(settings.database_url)
    prior = memory.recall("checkout cache latency")
    report = write_report("incident_commander/reports", "LOCAL-DEMO", {
        "timeline": "Alert received → evidence collected → proposal reviewed",
        "evidence": "Checkout error rate increased after cache release",
        "decision": "Rollback proposed; external execution not attempted",
        "approver": "pending",
        "outcome": "awaiting approval",
    })
    memory.remember("LOCAL-DEMO", "checkout cache latency", "rollback proposed")
    print(f"report={report} prior_incidents={len(prior)}")


if __name__ == "__main__":
    main()
