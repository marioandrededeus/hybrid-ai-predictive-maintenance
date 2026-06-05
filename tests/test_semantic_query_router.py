from src.llm.semantic_query_router import (
    get_supported_query_examples,
    route_prompt_to_sql,
)


def test_blocks_out_of_scope_prompt():
    response = route_prompt_to_sql("What is the capital of France?")

    assert response["status"] == "blocked"
    assert response["intent"] is None
    assert response["sql"] is None


def test_blocks_generic_vibration_outside_project_context():
    response = route_prompt_to_sql("Vibration level of Etna vulcan")

    assert response["status"] == "blocked"
    assert response["intent"] is None
    assert response["sql"] is None


def test_matches_average_anomaly_score_by_scenario():
    response = route_prompt_to_sql("Show average anomaly score by scenario")

    assert response["status"] == "matched"
    assert response["intent"] == "average_anomaly_score_by_scenario"
    assert "SELECT" in response["sql"]


def test_matches_lubrication_issues():
    response = route_prompt_to_sql("Show lubrication issues")

    assert response["status"] == "matched"
    assert response["intent"] == "lubrication_issues"
    assert "carpet_lubrication_issue" in response["sql"]


def test_matches_structural_looseness_cases():
    response = route_prompt_to_sql("Show structural looseness cases")

    assert response["status"] == "matched"
    assert response["intent"] == "structural_looseness_cases"
    assert "structural_looseness" in response["sql"]


def test_valid_domain_prompt_without_template_returns_not_matched():
    response = route_prompt_to_sql(
        "Explain maintenance anomaly behavior for vibration sensors"
    )

    assert response["status"] == "not_matched"
    assert response["intent"] is None
    assert response["sql"] is None


def test_supported_query_examples_are_available():
    examples = get_supported_query_examples()

    assert isinstance(examples, list)
    assert len(examples) > 0
    assert "average anomaly score" in examples

def test_matches_high_severity_diagnostics_short_prompt():
    from src.llm.semantic_query_router import route_prompt_to_sql

    result = route_prompt_to_sql("show high severity diagnostics")

    assert result["status"] == "matched"
    assert result["intent"] == "high_severity_diagnostics"
    assert result["sql"] is not None
    assert "severity_level" in result["sql"]
    assert "LOWER(s.severity_level) = 'high'" in result["sql"]


def test_matches_high_severity_diagnostics_explicit_prompt():
    from src.llm.semantic_query_router import route_prompt_to_sql

    result = route_prompt_to_sql(
        "show high severity predictive maintenance diagnostics for vibration measurements"
    )

    assert result["status"] == "matched"
    assert result["intent"] == "high_severity_diagnostics"
    assert result["sql"] is not None
    assert "vibration_measurements" in result["sql"]