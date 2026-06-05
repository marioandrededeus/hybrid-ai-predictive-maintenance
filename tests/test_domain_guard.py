from src.llm.domain_guard import get_domain_guard_response, is_prompt_in_domain


def test_blocks_empty_prompt():
    assert is_prompt_in_domain("") is False


def test_blocks_out_of_scope_prompt():
    response = get_domain_guard_response("What is the capital of France?")

    assert response["is_allowed"] is False
    assert response["reason"] == "Prompt is outside the predictive maintenance domain."


def test_blocks_generic_vibration_outside_industrial_context():
    response = get_domain_guard_response("Vibration level of Etna vulcan")

    assert response["is_allowed"] is False


def test_allows_anomaly_score_by_scenario():
    response = get_domain_guard_response("Show anomaly score by scenario")

    assert response["is_allowed"] is True


def test_allows_predictive_maintenance_risk_question():
    response = get_domain_guard_response(
        "Which maintenance scenarios have the highest anomaly risk?"
    )

    assert response["is_allowed"] is True


def test_allows_human_validation_question():
    response = get_domain_guard_response(
        "Which anomaly measurements require human validation?"
    )

    assert response["is_allowed"] is True