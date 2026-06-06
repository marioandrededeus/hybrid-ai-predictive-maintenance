from src.llm.semantic_query_router import (
    get_supported_demo_questions,
    get_supported_query_examples,
    route_prompt_to_sql,
)

def test_supported_demo_questions_are_available():
    demo_questions = get_supported_demo_questions()

    assert isinstance(demo_questions, list)
    assert len(demo_questions) > 0

    first_question = demo_questions[0]

    assert "category" in first_question
    assert "english" in first_question
    assert "portuguese" in first_question
    assert "spanish" in first_question

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


def test_valid_domain_prompt_without_template_returns_not_matched(monkeypatch):
    from src.llm import embedding_router

    def fake_route_prompt_by_embedding(prompt: str) -> dict:
        return {
            "status": "not_matched",
            "intent": None,
            "sql": None,
            "similarity_score": 0.31,
            "matched_example": None,
            "routing_method": "embedding_similarity",
        }

    monkeypatch.setattr(
        embedding_router,
        "route_prompt_by_embedding",
        fake_route_prompt_by_embedding,
    )

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
    assert "anomaly score by scenario" in examples
    assert "average rms velocity by predictive maintenance scenario" in examples
    assert "which assets have the highest anomaly risk" in examples

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

def test_route_prompt_uses_embedding_similarity_when_keyword_does_not_match(monkeypatch):
    from src.llm import embedding_router

    def fake_route_prompt_by_embedding(prompt: str) -> dict:
        return {
            "status": "matched",
            "intent": "highest_risk_assets",
            "sql": "SELECT 1;",
            "similarity_score": 0.82,
            "matched_example": "which assets have the highest anomaly risk",
            "routing_method": "embedding_similarity",
        }

    monkeypatch.setattr(
        embedding_router,
        "route_prompt_by_embedding",
        fake_route_prompt_by_embedding,
    )

    response = route_prompt_to_sql(
        "which predictive maintenance machines look most critical?"
    )

    assert response["status"] == "matched"
    assert response["intent"] == "highest_risk_assets"
    assert response["sql"] == "SELECT 1;"
    assert response["routing_method"] == "embedding_similarity"
    assert response["similarity_score"] == 0.82
    assert response["matched_example"] == "which assets have the highest anomaly risk"