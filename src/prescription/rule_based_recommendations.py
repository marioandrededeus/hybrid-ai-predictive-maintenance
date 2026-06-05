"""
Rule-based recommendation module for the Hybrid AI Predictive Maintenance project.

This module creates initial maintenance recommendations based on scenario labels,
spectral indicators, anomaly scores, and vibration behavior.

It is intentionally simple and transparent.
Later, this layer will be combined with RAG and LLM-based reasoning.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import get_full_diagnostic_view  # noqa: E402


def classify_action_priority(anomaly_score: float) -> str:
    """Classify action priority based on anomaly score."""
    if anomaly_score < 0.3:
        return "Low"
    if anomaly_score < 0.7:
        return "Medium"

    return "High"


def recommend_for_normal_operation(row: pd.Series) -> dict:
    """Generate recommendation for normal operation."""
    return {
        "measurement_id": row["measurement_id"],
        "scenario_label": row["scenario_label"],
        "priority": "Low",
        "recommended_action": "Continue routine monitoring.",
        "technical_reason": (
            "The vibration pattern shows low anomaly score and stable spectral behavior."
        ),
        "human_validation_required": False,
    }


def recommend_for_lubrication_issue(row: pd.Series) -> dict:
    """Generate recommendation for possible lubrication-related carpet pattern."""
    priority = classify_action_priority(row["anomaly_score"])

    return {
        "measurement_id": row["measurement_id"],
        "scenario_label": row["scenario_label"],
        "priority": priority,
        "recommended_action": (
            "Inspect lubrication condition, verify lubricant level and contamination, "
            "and schedule a bearing inspection if the pattern persists."
        ),
        "technical_reason": (
            "The scenario shows elevated broadband and high-frequency energy, "
            "which may indicate a carpet pattern associated with possible lubrication issues."
        ),
        "human_validation_required": True,
    }


def recommend_for_structural_looseness(row: pd.Series) -> dict:
    """Generate recommendation for possible structural looseness."""
    priority = classify_action_priority(row["anomaly_score"])

    return {
        "measurement_id": row["measurement_id"],
        "scenario_label": row["scenario_label"],
        "priority": priority,
        "recommended_action": (
            "Inspect foundation, bolts, supports, coupling alignment, and mechanical clearances."
        ),
        "technical_reason": (
            "The scenario shows increased low-frequency energy, harmonic behavior, "
            "and subharmonic components, which may indicate structural looseness."
        ),
        "human_validation_required": True,
    }


def recommend_for_unknown_pattern(row: pd.Series) -> dict:
    """Generate fallback recommendation for unknown abnormal patterns."""
    priority = classify_action_priority(row["anomaly_score"])

    return {
        "measurement_id": row["measurement_id"],
        "scenario_label": row["scenario_label"],
        "priority": priority,
        "recommended_action": (
            "Review the vibration signature with a maintenance specialist before taking action."
        ),
        "technical_reason": (
            "The vibration behavior does not match a known rule-based scenario."
        ),
        "human_validation_required": True,
    }


def generate_recommendation(row: pd.Series) -> dict:
    """Generate a recommendation for one measurement."""
    scenario_name = str(row["scenario_name"]).lower()
    scenario_label = str(row["scenario_label"]).lower()

    if "normal" in scenario_name or "normal" in scenario_label:
        return recommend_for_normal_operation(row)

    if "lubrication" in scenario_name or "carpet" in scenario_name:
        return recommend_for_lubrication_issue(row)

    if "lubrication" in scenario_label or "carpet" in scenario_label:
        return recommend_for_lubrication_issue(row)

    if "looseness" in scenario_name or "looseness" in scenario_label:
        return recommend_for_structural_looseness(row)

    return recommend_for_unknown_pattern(row)


def generate_recommendations(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate rule-based recommendations for all measurements."""
    if df is None:
        df = get_full_diagnostic_view()

    recommendations = [
        generate_recommendation(row)
        for _, row in df.iterrows()
    ]

    return pd.DataFrame(recommendations)


def main() -> None:
    """Run a quick manual validation."""
    recommendations_df = generate_recommendations()

    print("Rule-based recommendations")
    print(recommendations_df)


if __name__ == "__main__":
    main()