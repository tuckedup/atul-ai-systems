import sqlite3
from pathlib import Path

from evalops.oracles import (
    code_oracle,
    extract_oracle,
    final_answer,
    parse_bfcl_ground_truth,
    reason_oracle,
    sql_oracle,
    tool_use_oracle,
)

# ---------------------------------------------------------------- final_answer


def test_final_answer_takes_the_last_number_not_an_earlier_intermediate_one():
    # Regression test for the bug the module docstring calls out: `re.search` returns the
    # FIRST match, so an early "= 19.50" in a step-by-step solution was returned instead of
    # the stated final answer, mislabelling ~130/150 correct GSM8K responses as wrong.
    text = (
        "Step 1: subtotal is 19.50 after tax.\n"
        "Step 2: apply the discount.\n"
        "Final answer: 26"
    )
    assert final_answer(text) == "26"


def test_final_answer_hash_marker():
    assert final_answer("reasoning...\n#### 42") == "42"


def test_final_answer_bold_on_last_line():
    assert final_answer("The computation yields\n**243**") == "243"


def test_final_answer_latex_decorated_unit():
    assert final_answer(r"The mass comes out to 48 \text{ g}") == "48"


def test_final_answer_dollar_amount_with_comma():
    assert final_answer("Total cost: $1,234") == "1234"


def test_final_answer_none_when_no_number_present():
    assert final_answer("there is no numeric content in this response at all") is None


# ---------------------------------------------------------------- reason_oracle


def test_reason_oracle_correct_match():
    r = reason_oracle("The final answer is 26", gold="26")
    assert r.label == 1


def test_reason_oracle_numeric_tolerance():
    r = reason_oracle("26", gold="26.0")
    assert r.label == 1


def test_reason_oracle_comma_handling():
    r = reason_oracle("1,234", gold="1234")
    assert r.label == 1


def test_reason_oracle_wrong_answer_is_zero():
    r = reason_oracle("the answer is 100", gold="99")
    assert r.label == 0


# ---------------------------------------------------------------- code_oracle


def test_code_oracle_correct_implementation_passes():
    cand = "def add(a, b):\n    return a + b\n"
    test_program = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    r = code_oracle(cand, test_program=test_program)
    assert r.label == 1


def test_code_oracle_wrong_implementation_fails():
    cand = "def add(a, b):\n    return a - b\n"
    test_program = "assert add(2, 3) == 5\n"
    r = code_oracle(cand, test_program=test_program)
    assert r.label == 0


def test_code_oracle_syntax_error_is_a_decided_fail():
    cand = "def add(a, b:\n    return a + b\n"
    r = code_oracle(cand, test_program="pass")
    assert r.label == 0
    assert "syntax error" in r.detail


def test_code_oracle_infinite_loop_fails_via_timeout():
    cand = "while True:\n    pass\n"
    r = code_oracle(cand, test_program="pass", timeout_s=5)
    assert r.label == 0
    assert "did not terminate" in r.detail


def test_code_oracle_runs_outside_the_repo_root():
    # A candidate that tries to read a file only present at the repo root (pyproject.toml)
    # must not find it: the subprocess runs in a per-case temp directory, never repo root.
    cand = (
        "def check():\n"
        "    try:\n"
        "        open('pyproject.toml')\n"
        "        return True\n"
        "    except FileNotFoundError:\n"
        "        return False\n"
    )
    test_program = "assert check() is False\n"
    r = code_oracle(cand, test_program=test_program)
    assert r.label == 1  # the assertion that the file is NOT found passed


# ---------------------------------------------------------------- extract_oracle


GOLD = {"a": 1, "b": "x"}


def test_extract_oracle_key_order_irrelevant():
    assert extract_oracle('{"b":"x","a":1}', gold=GOLD).label == 1


def test_extract_oracle_int_float_string_numbers_equal():
    assert extract_oracle('{"a":1.0,"b":"x"}', gold=GOLD).label == 1
    assert extract_oracle('{"a":"1","b":"x"}', gold=GOLD).label == 1


def test_extract_oracle_markdown_fenced_json_parses():
    text = '```json\n{"a":1,"b":"x"}\n```'
    assert extract_oracle(text, gold=GOLD).label == 1


def test_extract_oracle_missing_field_fails():
    assert extract_oracle('{"b":"x"}', gold=GOLD).label == 0


def test_extract_oracle_extra_field_fails():
    assert extract_oracle('{"a":1,"b":"x","c":2}', gold=GOLD).label == 0


def test_extract_oracle_unparseable_is_zero():
    assert extract_oracle("not json at all", gold=GOLD).label == 0


# ---------------------------------------------------------------- tool_use_oracle


BFCL_GT_RAW = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def test_parse_bfcl_ground_truth_and_pass_with_unit_supplied():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    assert name == "calculate_triangle_area"
    cand = '{"name":"calculate_triangle_area","arguments":{"base":10,"height":5,"unit":"units"}}'
    r = tool_use_oracle(cand, gold_name=name, gold_args=args)
    assert r.label == 1


def test_parse_bfcl_ground_truth_optional_argument_may_be_omitted():
    # "" in the accepted-values list marks `unit` optional per the BFCL convention.
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = '{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}'
    r = tool_use_oracle(cand, gold_name=name, gold_args=args)
    assert r.label == 1


def test_tool_use_oracle_wrong_function_name_fails():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = '{"name":"other_function","arguments":{"base":10,"height":5,"unit":"units"}}'
    assert tool_use_oracle(cand, gold_name=name, gold_args=args).label == 0


def test_tool_use_oracle_wrong_argument_value_fails():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = '{"name":"calculate_triangle_area","arguments":{"base":11,"height":5,"unit":"units"}}'
    assert tool_use_oracle(cand, gold_name=name, gold_args=args).label == 0


def test_tool_use_oracle_missing_required_argument_fails():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = '{"name":"calculate_triangle_area","arguments":{"height":5,"unit":"units"}}'
    assert tool_use_oracle(cand, gold_name=name, gold_args=args).label == 0


def test_tool_use_oracle_invented_argument_fails():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = (
        '{"name":"calculate_triangle_area",'
        '"arguments":{"base":10,"height":5,"unit":"units","extra_arg":42}}'
    )
    assert tool_use_oracle(cand, gold_name=name, gold_args=args).label == 0


def test_tool_use_oracle_accepts_python_call_syntax():
    name, args = parse_bfcl_ground_truth(BFCL_GT_RAW)
    cand = "calculate_triangle_area(base=10, height=5, unit='units')"
    assert tool_use_oracle(cand, gold_name=name, gold_args=args).label == 1


# ---------------------------------------------------------------- sql_oracle


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, cat TEXT)")
    conn.executemany(
        "INSERT INTO t VALUES (?,?,?)", [(1, "a", "x"), (2, "b", "y"), (3, "c", "x")]
    )
    conn.commit()
    conn.close()
    return db


def test_sql_oracle_equivalent_but_differently_written_queries_match(tmp_path):
    db = _make_db(tmp_path)
    gold_sql = "SELECT name FROM t WHERE cat = 'x'"
    # Different alias / IN vs OR, same result set.
    cand_sql = "SELECT t.name FROM t WHERE cat IN ('x')"
    r = sql_oracle(cand_sql, gold_sql=gold_sql, db_path=db)
    assert r.label == 1


def test_sql_oracle_wrong_filter_fails(tmp_path):
    db = _make_db(tmp_path)
    gold_sql = "SELECT name FROM t WHERE cat = 'x'"
    cand_sql = "SELECT name FROM t WHERE cat = 'y'"
    r = sql_oracle(cand_sql, gold_sql=gold_sql, db_path=db)
    assert r.label == 0


def test_sql_oracle_broken_candidate_query_is_zero(tmp_path):
    db = _make_db(tmp_path)
    gold_sql = "SELECT name FROM t WHERE cat = 'x'"
    r = sql_oracle("not sql at all !!", gold_sql=gold_sql, db_path=db)
    assert r.label == 0


def test_sql_oracle_nonexistent_db_is_undecided(tmp_path):
    gold_sql = "SELECT name FROM t WHERE cat = 'x'"
    r = sql_oracle("SELECT name FROM t", gold_sql=gold_sql, db_path=tmp_path / "missing.db")
    assert r.label is None
    assert r.decided is False


def test_sql_oracle_broken_gold_query_is_undecided(tmp_path):
    db = _make_db(tmp_path)
    r = sql_oracle("SELECT name FROM t", gold_sql="SELECT * FROM nonexistent_table", db_path=db)
    assert r.label is None
