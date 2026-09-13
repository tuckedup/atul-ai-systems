from incident_commander.agents.scoping import SCOPES, prompt, scoped_context


def test_every_node_receives_only_allowed_context():
    all_context = {key: f"secret-{key}" for values in SCOPES.values() for key in values}
    all_context["source_code"] = "must never leak to logs"
    for node, allowed in SCOPES.items():
        visible = scoped_context(node, all_context)
        assert set(visible) == allowed
        assert "source_code" not in prompt(node, all_context)
    assert "relevant_files" not in scoped_context("logs_agent", all_context)

