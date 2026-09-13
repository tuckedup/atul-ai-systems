from incident_commander.memory.long_term import LongTermMemory
from incident_commander.security import redact


def test_pii_is_redacted_before_models():
    text = "owner jane@example.com bearer abc.def token=secret123"
    cleaned = redact(text)
    assert "jane@example.com" not in cleaned and "abc.def" not in cleaned and "secret123" not in cleaned


def test_long_term_memory_recalls_prior_incident(tmp_path):
    memory = LongTermMemory(f"sqlite:///{tmp_path}/memory.db")
    memory.remember("INC-1", "checkout latency after cache deploy", "rolled back cache")
    result = memory.recall("cache latency")
    assert result[0]["id"] == "INC-1" and "rolled back" in result[0]["outcome"]
