from __future__ import annotations

from pathlib import Path

import pytest
from aisys.evals import EvalSuite

CASE_YAML = """\
- id: case-1
  input: hello
  expected: hello
  grader: exact
- id: case-2
  input: world
  expected: world
  grader: exact
"""


def test_load_single_file_loads_all_cases(tmp_path: Path) -> None:
    """Regression test for defect 2: a single suite file used to silently load zero cases."""
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(CASE_YAML, encoding="utf-8")
    suite = EvalSuite.load(suite_file)
    assert len(suite.cases) == 2
    assert [c.id for c in suite.cases] == ["case-1", "case-2"]


def test_load_directory_still_recurses(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.yaml").write_text("- {id: a1, input: x, grader: exact}\n", encoding="utf-8")
    (tmp_path / "sub" / "b.yml").write_text("- {id: b1, input: y, grader: exact}\n", encoding="utf-8")
    suite = EvalSuite.load(tmp_path)
    assert sorted(c.id for c in suite.cases) == ["a1", "b1"]


def test_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EvalSuite.load(tmp_path / "does-not-exist.yaml")


def test_empty_yaml_list_raises_value_error(tmp_path: Path) -> None:
    suite_file = tmp_path / "empty.yaml"
    suite_file.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero cases"):
        EvalSuite.load(suite_file)


def test_duplicate_ids_raise_value_error_naming_the_duplicate(tmp_path: Path) -> None:
    suite_file = tmp_path / "dupes.yaml"
    suite_file.write_text(
        "- {id: dup, input: x, grader: exact}\n- {id: dup, input: y, grader: exact}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dup"):
        EvalSuite.load(suite_file)


def test_unknown_grader_raises_value_error(tmp_path: Path) -> None:
    suite_file = tmp_path / "bad_grader.yaml"
    suite_file.write_text("- {id: c1, input: x, grader: not_a_real_grader}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_real_grader"):
        EvalSuite.load(suite_file)


def test_unknown_grader_allowed_when_not_strict(tmp_path: Path) -> None:
    suite_file = tmp_path / "bad_grader.yaml"
    suite_file.write_text("- {id: c1, input: x, grader: not_a_real_grader}\n", encoding="utf-8")
    suite = EvalSuite.load(suite_file, strict=False)
    assert len(suite.cases) == 1
