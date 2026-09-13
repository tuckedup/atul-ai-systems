from gateway.reliability import (
    AdmissionController,
    CircuitBreaker,
    TenantBudget,
    TokenBucket,
)


def test_circuit_opens_and_budget_fails_closed():
    breaker = CircuitBreaker(failure_threshold=2, reset_s=60)
    breaker.failure()
    assert not breaker.is_open
    breaker.failure()
    assert breaker.is_open
    budget = TenantBudget(1.0)
    assert budget.charge(0.75)
    assert not budget.charge(0.26)


def test_rate_and_admission_limits():
    bucket = TokenBucket(rate=0, capacity=1)
    assert bucket.allow() and not bucket.allow()
    admission = AdmissionController(1)
    assert admission.acquire() and not admission.acquire()
    admission.release()
