# Fixtures — NOT calibration data

Everything in this directory is machine-generated and exists only to exercise code paths. None
of it is admissible as evaluation or calibration evidence, and the loaders refuse it by
provenance rather than by convention.

## `synthetic_selfagreement_labels.csv`

Produced by the old `generate_data.py`:

```python
for number in range(50):
    value = number % 2
    labels.append(f"label-{number + 1:02d},{value},{value}")
    #                                       human  judge   <- same variable
```

`human_score` and `judge_score` were written from one variable, so Cohen's κ on this file is
1.0 by construction and measures nothing. It is kept only as the negative fixture for
`test_calibration_gate.py`, which asserts that a fixture-provenance label is **rejected**.

Real labels live in `../data/annotations.jsonl` with an explicit
`LabelProvenance` of `human_expert_annotation`, `human_local_annotation` or
`human_gold_reference_oracle`.

## `generated_router_smoke.yaml`

200 cases of the form `input: "coding case 7"`, `expected: "ok"`, `grader: exact`. There is no
task content, so a success rate over it is not a quality measurement. Kept as a loader/router
smoke fixture. The real corpus is built by `evalops/build.py` from public benchmark sources —
see `../data/raw/MANIFEST.json` for provenance.
