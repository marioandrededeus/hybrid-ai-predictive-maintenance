from src.database.queries import get_scenario_by_id, get_measurements_by_scenario
from src.analysis.scenario_summary import generate_scenario_summaries
from src.prescription.rule_based_recommendations import generate_recommendations


def run_maintenance_agent(scenario_id: int) -> dict:
    """
    Simple deterministic maintenance agent.

    This agent does not use a real LLM yet.
    It orchestrates existing project tools:
    - scenario retrieval
    - vibration measurements retrieval
    - scenario summary
    - rule-based recommendations
    """

    scenario = get_scenario_by_id(scenario_id)

    if scenario is None:
        return {
            "status": "error",
            "message": f"Scenario ID {scenario_id} not found.",
            "scenario_id": scenario_id,
            "summary": None,
            "recommendations": [],
            "agent_reasoning": [
                "The agent tried to retrieve the scenario from the database.",
                "No matching scenario was found.",
                "No diagnostic or recommendation was generated."
            ]
        }

    measurements_df = get_measurements_by_scenario(scenario["scenario_name"])

    summaries_df = generate_scenario_summaries()

    if "scenario_id" in summaries_df.columns:
        filtered_summary_df = summaries_df[summaries_df["scenario_id"] == scenario_id]
    
    elif "scenario_name" in summaries_df.columns:
        filtered_summary_df = summaries_df[
            summaries_df["scenario_name"] == scenario["scenario_name"]
        ]
    elif "scenario_label" in summaries_df.columns:
        filtered_summary_df = summaries_df[
            summaries_df["scenario_label"] == scenario["scenario_label"]
        ]
    else:
        filtered_summary_df = summaries_df.head(0)

    if filtered_summary_df.empty:
        summary = None
    else:
        summary = filtered_summary_df.iloc[0].to_dict()

    recommendations = generate_recommendations(measurements_df)

    agent_reasoning = [
        "The agent retrieved the selected industrial scenario from the database.",
        "The agent retrieved vibration measurements related to the scenario.",
        "The agent generated a technical summary using the scenario data.",
        "The agent applied rule-based maintenance logic to generate recommendations.",
        "No real LLM was used in this version."
    ]

    return {
        "status": "success",
        "message": "Maintenance agent executed successfully.",
        "scenario_id": scenario_id,
        "scenario": scenario,
        "measurements_count": len(measurements_df),
        "summary": summary,
        "recommendations": recommendations,
        "agent_reasoning": agent_reasoning
    }