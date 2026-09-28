# Run every target from the WORKTREE ROOT, not from routebench/: the .env that carries provider
# credentials is only resolved from the root, and evalops imports need routebench on PYTHONPATH.
PY := ../.venv/Scripts/python.exe
ROOT := cd .. && PYTHONPATH=routebench .venv/Scripts/python.exe

.PHONY: test lint bench corpus splits calibrate-dev calibrate-freeze calibrate-test calibrate-check annotate-export annotate annotate-status

test:  ; $(ROOT) -m pytest routebench/tests -q --basetemp .local/pytest-routebench
lint:  ; cd .. && .venv/Scripts/python.exe -m ruff check routebench/gateway routebench/evalops routebench/tests

## Print the measured quality x cost x latency matrix. Anything without an artifact on disk
## prints "not measured" rather than a placeholder.
bench: ; $(ROOT) -m evalops.bench

## --- corpus ---------------------------------------------------------------
## Rebuild the calibration corpus from the fetched public sources. Generation is cached, so a
## re-run costs nothing for candidates already produced.
corpus: ; $(ROOT) -m evalops.build all --per-class 260 --families reason,code,code_mbpp,sql,tool_use
splits: ; $(ROOT) -c "from evalops.dataset import Corpus; from evalops import splits; c=Corpus.load('routebench/evalops/data'); sp=splits.plan(c.cases); print(splits.verify(c.cases, sp)); sp.save('routebench/evalops/data/splits.json')"

## --- calibration ----------------------------------------------------------
## Strict order. `test` refuses to run before `freeze`, and refuses to re-select anything.
calibrate-dev:    ; $(ROOT) -m evalops.run_calibration dev --budget 25
calibrate-freeze: ; $(ROOT) -m evalops.run_calibration freeze
calibrate-test:   ; $(ROOT) -m evalops.run_calibration test --budget 10
## Validate the frozen artifact the way CI and the promotion gate do. Exits non-zero on failure.
calibrate-check:  ; $(ROOT) -m evalops.calibrate routebench/evalops/data/calibration_bundle.json

## --- human annotation -----------------------------------------------------
annotate-export: ; $(ROOT) -m evalops.annotate export --n 100
annotate:        ; $(ROOT) -m evalops.annotate label --annotator $(ANNOTATOR)
annotate-status: ; $(ROOT) -m evalops.annotate status
