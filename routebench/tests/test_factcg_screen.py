import pytest
from evalops.factcg_screen import source_chunks, validate_scores


def test_chunks_preserve_content_and_keep_oversized_sentence():
    sentences = ["A short sentence.", "A longer sentence.", "The last sentence."]
    text = "\n".join(sentences)
    assert source_chunks(text, sentences, [10, 600, 10]) == sentences


def test_chunks_detect_dropped_source_content():
    with pytest.raises(ValueError, match="lost source"):
        source_chunks("First. Second.", ["First."], [1])


@pytest.mark.parametrize("rows", [[], [{"id": "a", "score": float("nan")}],
                                  [{"id": "a", "score": 1.1}], [{"id": "b", "score": 0.3}]])
def test_invalid_population_or_score_cannot_be_reported(rows):
    with pytest.raises(ValueError):
        validate_scores(["a"], rows)


def test_duplicate_score_does_not_replace_missing_case():
    with pytest.raises(ValueError, match="duplicate"):
        validate_scores(["a", "b"], [{"id": "a", "score": 0.2}, {"id": "a", "score": 0.4}])


def test_valid_zero_and_one_scores():
    assert validate_scores(["a", "b"], [{"id": "b", "score": 1}, {"id": "a", "score": 0}]) == {"b": 1, "a": 0}
