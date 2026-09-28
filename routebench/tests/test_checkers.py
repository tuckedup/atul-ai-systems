"""Tests for the MiniCheck / AlignScore adapters.

Every test here runs with no torch, no transformers, no nltk, no weights, no network and no GPU.
That is the point: a suite that only runs after a multi-gigabyte download is a suite that does not
run, and the properties worth checking -- segmentation, aggregation, error propagation, revision
binding, cache separation -- are all independent of the neural network.

`test_no_heavy_imports_at_module_load` is the test that keeps that true.
"""
from __future__ import annotations

import sys

import pytest
from evalops.checkers import (
    AGGREGATIONS,
    DEFAULT_SPECS,
    UPSTREAM,
    CheckerError,
    CheckerResult,
    CheckerSpec,
    FakeSupportChecker,
    aggregate_sentence_scores,
    check_case,
    load_checker,
    split_claim_sentences,
)
from evalops.dataset import CaseRecord

SOURCE = (
    "Rizespor confirmed the loan on Monday. The fee was undisclosed. "
    "The club is considering a permanent deal next season."
)


def _case(candidate: str = "He joined on loan. The fee was undisclosed.", context: str = SOURCE):
    return CaseRecord(
        case_id="gnd-1", group_id="doc-1", task_class="summarize",
        task_input="Summarise the article.", context=context, reference="",
        candidate_output=candidate,
    )


def _spec(**kw) -> CheckerSpec:
    base = {"checker_id": "test-checker", "backend": "minicheck", "model_name": "flan-t5-large"}
    base.update(kw)
    return CheckerSpec(**base)  # type: ignore[arg-type]


#: A deterministic sentence splitter, so segmentation is fixed without importing nltk.
def _split(text: str) -> list[str]:
    return [p.strip() + "." for p in text.split(".") if p.strip()]


# ---------------------------------------------------------------- the lazy-import contract


def test_no_heavy_imports_at_module_load():
    # If this fails, the whole suite becomes undiscoverable on a machine without torch.
    heavy = {"torch", "transformers", "nltk", "vllm", "minicheck", "alignscore"}
    assert not (heavy & set(sys.modules)), heavy & set(sys.modules)


def test_load_checker_refuses_before_the_weights_licence_is_checked():
    # The code licences say nothing about the weights; MiniCheck asks for separate commercial
    # licensing for its 7B model. A default that downloaded weights would decide that silently.
    with pytest.raises(CheckerError, match="weights_licence_checked"):
        load_checker(_spec())


def test_load_checker_refuses_an_llm_backed_variant_without_approval():
    spec = _spec(model_name="Bespoke-MiniCheck-7B", weights_licence_checked=True)
    with pytest.raises(CheckerError, match="commercial licence"):
        load_checker(spec)


def test_load_checker_refuses_alignscore_without_a_checkpoint():
    spec = _spec(checker_id="a", backend="alignscore", model_name="roberta-base",
                 weights_licence_checked=True)
    # The policy check must fire before the import check, so the message is about the checkpoint
    # and not about whichever package happens to be missing.
    with pytest.raises(CheckerError, match="ckpt_path"):
        load_checker(spec)


def test_unknown_backend_is_refused():
    with pytest.raises(CheckerError, match="unknown backend"):
        _ = _spec(backend="vibes").upstream


# ---------------------------------------------------------------- upstream revision binding


def test_every_default_spec_pins_an_upstream_commit():
    for spec in DEFAULT_SPECS:
        assert len(spec.upstream["commit"]) == 40, spec.checker_id
        assert spec.upstream["code_licence"], spec.checker_id


def test_the_pinned_commits_match_the_reproduction_pack():
    assert UPSTREAM["minicheck"]["commit"] == "b58b9fa69acbd1015ec970fa65dd752413a053d2"
    assert UPSTREAM["alignscore"]["commit"] == "a0936d5afee642a46b22f6c02a163478447aa493"


def test_the_upstream_commit_is_part_of_the_config_hash():
    # A support score from a different revision is a different measurement. A cache that reused it
    # across revisions would attribute one model's numbers to another.
    spec = _spec()
    before = spec.config_hash
    UPSTREAM["minicheck"]["commit"] = "0" * 40
    try:
        assert _spec().config_hash != before
    finally:
        UPSTREAM["minicheck"]["commit"] = "b58b9fa69acbd1015ec970fa65dd752413a053d2"
    assert _spec().config_hash == before


@pytest.mark.parametrize("field,value", [
    ("model_name", "roberta-large"),
    ("chunk_size", 400),
    ("split_sentences", False),
    ("aggregation", "mean"),
    ("evaluation_mode", "bin_sp"),
])
def test_anything_that_changes_a_score_changes_the_config_hash(field, value):
    assert _spec(**{field: value}).config_hash != _spec().config_hash


@pytest.mark.parametrize("field,value", [
    ("checker_id", "renamed"),
    ("notes", "a different comment"),
    ("weights_licence_checked", True),
])
def test_labels_do_not_change_the_config_hash(field, value):
    assert _spec(**{field: value}).config_hash == _spec().config_hash


def test_default_spec_ids_are_unique():
    ids = [s.checker_id for s in DEFAULT_SPECS]
    assert len(set(ids)) == len(ids)


def test_default_specs_have_distinct_config_hashes():
    hashes = [s.config_hash for s in DEFAULT_SPECS]
    assert len(set(hashes)) == len(hashes)


def test_the_default_minicheck_variant_is_not_the_licence_restricted_one():
    for spec in DEFAULT_SPECS:
        assert spec.model_name != "Bespoke-MiniCheck-7B"


# ---------------------------------------------------------------- aggregation (the adaptation)


def test_min_aggregation_lets_one_unsupported_sentence_sink_the_response():
    # Closest to the annotation question as written: "is EVERY claim supported".
    assert aggregate_sentence_scores([0.99, 0.99, 0.02], "min") == pytest.approx(0.02)


def test_mean_aggregation_tolerates_one_weak_sentence():
    assert aggregate_sentence_scores([0.9, 0.9, 0.3], "mean") == pytest.approx(0.7)


def test_supported_fraction_uses_upstreams_own_cutoff():
    assert aggregate_sentence_scores([0.9, 0.6, 0.4, 0.1], "supported_fraction") == pytest.approx(0.5)


def test_upstream_binary_reproduces_the_published_rule():
    assert aggregate_sentence_scores([0.51], "upstream_binary") == 1.0
    assert aggregate_sentence_scores([0.50], "upstream_binary") == 0.0


def test_every_declared_aggregation_is_callable_and_bounded():
    for name in AGGREGATIONS:
        value = aggregate_sentence_scores([0.2, 0.8], name)
        assert 0.0 <= value <= 1.0, name


def test_an_unknown_aggregation_is_refused():
    with pytest.raises(CheckerError, match="unknown aggregation"):
        aggregate_sentence_scores([0.5], "argmax-of-vibes")


def test_empty_scores_raise_rather_than_defaulting_to_zero():
    # 0.0 would enter the metric as an unsupported verdict for a case the checker never answered.
    with pytest.raises(CheckerError, match="error, not a score of zero"):
        aggregate_sentence_scores([], "min")


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_out_of_range_support_probabilities_are_refused(bad):
    with pytest.raises(CheckerError, match=r"\[0, 1\]"):
        aggregate_sentence_scores([0.5, bad], "min")


# ---------------------------------------------------------------- segmentation


def test_an_injected_splitter_is_recorded_as_such():
    # Substituting a splitter changes the segmentation and therefore the score, so it must never be
    # invisible in the record.
    result = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split)
    assert result.segmentation["splitter"] == "injected"
    assert result.segmentation["n_sentences"] == 2


def test_each_sentence_is_scored_against_the_full_source():
    checker = FakeSupportChecker(default=0.9)
    check_case(_case(), _spec(), checker, splitter=_split)
    (docs, claims), = checker.calls
    assert len(docs) == len(claims) == 2
    assert set(docs) == {SOURCE}, "the whole document is the premise for every sentence"


def test_sentence_scores_line_up_with_their_sentences():
    checker = FakeSupportChecker(scores={"He joined on loan.": 0.95,
                                         "The fee was undisclosed.": 0.10}, default=0.0)
    result = check_case(_case(), _spec(), checker, splitter=_split)
    assert result.sentences == ["He joined on loan.", "The fee was undisclosed."]
    assert result.sentence_scores == [0.95, 0.10]
    assert result.score == pytest.approx(0.10), "min aggregation"


def test_split_sentences_false_scores_the_whole_response_once():
    checker = FakeSupportChecker(default=0.8)
    result = check_case(_case(), _spec(split_sentences=False, aggregation="upstream_binary"),
                        checker, splitter=_split)
    (_docs, claims), = checker.calls
    assert len(claims) == 1
    assert claims[0] == "He joined on loan. The fee was undisclosed."
    assert result.score == 1.0
    assert result.segmentation["n_sentences"] == 1


def test_the_reference_and_task_input_are_not_part_of_the_premise():
    # The groundedness question is about the DOCUMENT. Widening the premise would let a claim be
    # "supported" by a gold answer the annotators never saw -- the same scoping error as the
    # evidence-location defect in judge.py, one layer out.
    case = CaseRecord(
        case_id="c", group_id="g", task_class="summarize", task_input="TASKINPUT",
        context=SOURCE, reference="REFERENCE-ANSWER", candidate_output="A sentence.",
    )
    checker = FakeSupportChecker(default=0.9)
    check_case(case, _spec(), checker, splitter=_split)
    (docs, _), = checker.calls
    assert "REFERENCE-ANSWER" not in docs[0]
    assert "TASKINPUT" not in docs[0]


def test_split_claim_sentences_needs_upstream_or_an_explicit_splitter():
    # Falling back to a different splitter would silently change the segmentation, so it refuses.
    if "minicheck" in sys.modules:  # pragma: no cover - upstream present in this environment
        pytest.skip("upstream MiniCheck is importable here")
    with pytest.raises(CheckerError, match="no longer upstream"):
        split_claim_sentences("One. Two.")


def test_an_injected_splitter_drops_blank_fragments():
    assert split_claim_sentences("One.  . Two.", splitter=lambda t: t.split(".")) == ["One", "Two"]


# ---------------------------------------------------------------- errors are errors


def test_an_empty_source_is_an_error_not_an_unsupported_verdict():
    result = check_case(_case(context="   "), _spec(), FakeSupportChecker(), splitter=_split)
    assert result.status == "empty_input"
    assert result.score is None
    assert result.usable is False
    assert "rather than scored 0.0" in (result.error or "")


def test_an_empty_response_is_an_error():
    result = check_case(_case(candidate="  "), _spec(), FakeSupportChecker(), splitter=_split)
    assert result.status == "empty_input"
    assert result.usable is False


def test_a_response_that_segments_to_nothing_is_an_error():
    result = check_case(_case(candidate="..."), _spec(), FakeSupportChecker(),
                        splitter=lambda t: [])
    assert result.status == "empty_input"
    assert result.usable is False


def test_an_oversized_source_is_refused_rather_than_truncated():
    # Upstream chunks long documents; truncating would change which sentences can support a claim
    # and the score would look perfectly normal.
    result = check_case(_case(context="x. " * 5000), _spec(), FakeSupportChecker(),
                        splitter=_split, max_source_chars=100)
    assert result.status == "invalid_output"
    assert "truncates" in (result.error or "")
    assert result.score is None


def test_a_checker_exception_becomes_a_recorded_error():
    result = check_case(_case(), _spec(), FakeSupportChecker(fail_on_call=1), splitter=_split)
    assert result.status == "checker_error"
    assert "scripted checker failure" in (result.error or "")
    assert result.score is None
    assert result.usable is False


def test_a_score_count_mismatch_is_an_error():
    class Broken:
        def score(self, docs, claims):
            return [0.5]  # one score for two sentences

    result = check_case(_case(), _spec(), Broken(), splitter=_split)
    assert result.status == "invalid_output"
    assert "for 2 sentences" in (result.error or "")


def test_an_out_of_range_score_from_the_checker_is_an_error():
    class Broken:
        def score(self, docs, claims):
            return [1.5] * len(claims)

    result = check_case(_case(), _spec(), Broken(), splitter=_split)
    assert result.status == "invalid_output"
    assert result.usable is False


def test_latency_is_recorded_even_on_failure():
    result = check_case(_case(), _spec(), FakeSupportChecker(fail_on_call=1), splitter=_split)
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------- the result record


def test_the_result_binds_the_exact_texts_it_scored():
    a = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split)
    b = check_case(_case(candidate="Something else entirely."), _spec(),
                   FakeSupportChecker(default=0.9), splitter=_split)
    assert a.source_hash == b.source_hash
    assert a.candidate_hash != b.candidate_hash


def test_the_result_carries_the_upstream_and_model_revision():
    result = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split)
    assert result.upstream_commit == UPSTREAM["minicheck"]["commit"]
    assert result.model_name == "flan-t5-large"
    assert result.config_hash == _spec().config_hash


def test_zero_cost_is_recorded_with_its_basis():
    # A bare 0.00 is indistinguishable from an unpriced model, which is how a cap stops binding.
    result = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split)
    assert result.cost_usd == 0.0
    assert result.cost_basis == "local_inference_no_provider_charge"


def test_the_trace_id_is_carried_through():
    result = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split,
                        trace_id="trace-123")
    assert result.trace_id == "trace-123"


def test_a_result_round_trips_through_dict():
    result = check_case(_case(), _spec(), FakeSupportChecker(default=0.9), splitter=_split)
    d = result.as_dict()
    assert d["checker_id"] == "test-checker"
    assert CheckerResult(**d).score == result.score


def test_usable_requires_both_ok_status_and_a_score():
    assert CheckerResult(case_id="c", checker_id="k", config_hash="h", backend="minicheck",
                         upstream_commit="x", model_name="m", score=0.5).usable is True
    assert CheckerResult(case_id="c", checker_id="k", config_hash="h", backend="minicheck",
                         upstream_commit="x", model_name="m", score=None).usable is False
    assert CheckerResult(case_id="c", checker_id="k", config_hash="h", backend="minicheck",
                         upstream_commit="x", model_name="m", score=0.5,
                         status="checker_error").usable is False


def test_two_aggregations_of_one_checker_are_separate_measurements():
    # Cache separation: the same model under a different aggregation must not share a cache slot.
    a = _spec(checker_id="a", aggregation="min")
    b = _spec(checker_id="b", aggregation="mean")
    assert a.config_hash != b.config_hash
    checker = FakeSupportChecker(scores={"He joined on loan.": 0.9,
                                         "The fee was undisclosed.": 0.1})
    ra = check_case(_case(), a, checker, splitter=_split)
    rb = check_case(_case(), b, checker, splitter=_split)
    assert ra.score == pytest.approx(0.1)
    assert rb.score == pytest.approx(0.5)


# ---------------------------------------------------------------- unsupported entities and numbers


def test_a_changed_number_is_scored_by_the_checker_not_by_the_adapter():
    # The adapter must not second-guess the model. Its job is segmentation, plumbing and honest
    # error handling; deciding whether "a record fee" contradicts "undisclosed" is the model's.
    case = _case(candidate="The fee was a record 20 million euros.")
    checker = FakeSupportChecker(scores={"The fee was a record 20 million euros.": 0.03})
    result = check_case(case, _spec(), checker, splitter=_split)
    assert result.score == pytest.approx(0.03)
    assert result.status == "ok"


def test_a_supported_paraphrase_is_scored_high():
    case = _case(candidate="Rizespor took him on loan.")
    checker = FakeSupportChecker(scores={"Rizespor took him on loan.": 0.97})
    assert check_case(case, _spec(), checker, splitter=_split).score == pytest.approx(0.97)
