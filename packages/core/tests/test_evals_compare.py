from aisys.evals import CaseResult, RunResult, calibrate, compare


def _run(name, scores, tag="bugfix"):
    return RunResult(name=name, results=[CaseResult(case_id=str(i), score=s, tags=[tag]) for i, s in enumerate(scores)])


def test_compare_blocks_on_regression():
    base = _run("base", [1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    cand = _run("cand", [1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert compare(base, cand).blocked is True
    assert compare(base, base).blocked is False


def test_calibrate_reports_kappa():
    out = calibrate([1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1])
    assert 0 <= out["agreement"] <= 1 and "kappa" in out
