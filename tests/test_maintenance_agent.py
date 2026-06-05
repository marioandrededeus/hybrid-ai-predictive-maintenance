from src.agents.maintenance_agent import run_maintenance_agent


def test_maintenance_agent_success_for_valid_scenario():
    response = run_maintenance_agent(1)

    assert response["status"] == "success"
    assert response["message"] == "Maintenance agent executed successfully."
    assert response["scenario_id"] == 1


def test_maintenance_agent_returns_scenario_data():
    response = run_maintenance_agent(1)

    assert response["scenario"] is not None
    assert response["scenario"]["scenario_id"] == 1
    assert response["scenario"]["scenario_name"] == "normal_operation"
    assert response["scenario"]["scenario_label"] == "Normal operation"


def test_maintenance_agent_returns_summary():
    response = run_maintenance_agent(1)

    assert response["summary"] is not None
    assert response["summary"]["scenario_label"] == "Normal operation"
    assert "risk_level" in response["summary"]


def test_maintenance_agent_returns_recommendations_as_records():
    response = run_maintenance_agent(3)

    assert isinstance(response["recommendations"], list)
    assert len(response["recommendations"]) > 0

    first_recommendation = response["recommendations"][0]

    assert isinstance(first_recommendation, dict)
    assert "measurement_id" in first_recommendation
    assert "recommended_action" in first_recommendation
    assert "technical_reason" in first_recommendation
    assert "human_validation_required" in first_recommendation


def test_maintenance_agent_returns_agent_reasoning():
    response = run_maintenance_agent(2)

    assert isinstance(response["agent_reasoning"], list)
    assert len(response["agent_reasoning"]) > 0
    assert any(
        "No real LLM was used" in step
        for step in response["agent_reasoning"]
    )


def test_maintenance_agent_error_for_invalid_scenario():
    response = run_maintenance_agent(999)

    assert response["status"] == "error"
    assert response["scenario_id"] == 999
    assert response["summary"] is None
    assert response["recommendations"] == []
    assert isinstance(response["agent_reasoning"], list)
