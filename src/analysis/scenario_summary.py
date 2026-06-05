"""
Scenario summary module for the Hybrid AI Predictive Maintenance project.

This module creates simple technical summaries from diagnostic data.
It prepares the project for later integration with LLMs, agents, RAG,
and human-in-the-loop validation.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import get_full_diagnostic_view  # noqa: E402


def classify_risk_level(avg_anomaly_score: float) -> str:
    """Classify risk level based on the average anomaly score."""
    if avg_anomaly_score < 0.3:
        return "Low"
    if avg_anomaly_score < 0.7:
        return "Medium"

    return "High"


def summarize_scenario(group: pd.DataFrame) -> dict:
    """Create a technical summary for a single scenario."""
    scenario_label = group["scenario_label"].iloc[0]
    severity_level = group["severity_level"].iloc[0]

    avg_anomaly_score = group["anomaly_score"].mean()
    avg_anomaly_probability = group["anomaly_probability"].mean()
    avg_rms_velocity = group["rms_velocity"].mean()
    avg_peak_velocity = group["peak_velocity"].mean()
    avg_broadband_energy = group["broadband_energy"].mean()
    avg_low_frequency_energy = group["low_frequency_energy"].mean()
    avg_high_frequency_energy = group["high_frequency_energy"].mean()
    avg_harmonic_ratio = group["harmonic_ratio"].mean()
    avg_subharmonic_ratio = group["subharmonic_ratio"].mean()

    risk_level = classify_risk_level(avg_anomaly_score)

    return {
        "scenario_label": scenario_label,
        "severity_level": severity_level,
        "measurement_count": len(group),
        "avg_anomaly_score": round(avg_anomaly_score, 3),
        "avg_anomaly_probability": round(avg_anomaly_probability, 3),
        "avg_rms_velocity": round(avg_rms_velocity, 3),
        "avg_peak_velocity": round(avg_peak_velocity, 3),
        "avg_broadband_energy": round(avg_broadband_energy, 3),
        "avg_low_frequency_energy": round(avg_low_frequency_energy, 3),
        "avg_high_frequency_energy": round(avg_high_frequency_energy, 3),
        "avg_harmonic_ratio": round(avg_harmonic_ratio, 3),
        "avg_subharmonic_ratio": round(avg_subharmonic_ratio, 3),
        "risk_level": risk_level,
    }


def generate_scenario_summaries(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate summaries for all available scenarios."""
    if df is None:
        df = get_full_diagnostic_view()

    summaries = []

    for _, group in df.groupby("scenario_label"):
        summaries.append(summarize_scenario(group))

    return pd.DataFrame(summaries)


def generate_text_summary(row: pd.Series) -> str:
    """Generate a simple readable text summary for one scenario."""
    scenario = row["scenario_label"]
    risk_level = row["risk_level"]
    anomaly_score = row["avg_anomaly_score"]
    rms_velocity = row["avg_rms_velocity"]
    broadband_energy = row["avg_broadband_energy"]
    low_frequency_energy = row["avg_low_frequency_energy"]
    high_frequency_energy = row["avg_high_frequency_energy"]
    harmonic_ratio = row["avg_harmonic_ratio"]

    if risk_level == "Low":
        interpretation = (
            "The scenario shows stable behavior, with low anomaly score "
            "and balanced vibration indicators."
        )
    elif "lubrication" in scenario.lower() or "carpet" in scenario.lower():
        interpretation = (
            "The scenario shows elevated broadband and high-frequency energy, "
            "which is consistent with a carpet pattern associated with possible "
            "lubrication issues."
        )
    elif "looseness" in scenario.lower():
        interpretation = (
            "The scenario shows increased low-frequency energy and harmonic behavior, "
            "which is consistent with possible structural looseness."
        )
    else:
        interpretation = (
            "The scenario presents abnormal vibration behavior and should be reviewed "
            "by a maintenance specialist."
        )

    return (
        f"Scenario: {scenario}\n"
        f"Risk level: {risk_level}\n"
        f"Average anomaly score: {anomaly_score}\n"
        f"Average RMS velocity: {rms_velocity}\n"
        f"Average broadband energy: {broadband_energy}\n"
        f"Average low-frequency energy: {low_frequency_energy}\n"
        f"Average high-frequency energy: {high_frequency_energy}\n"
        f"Average harmonic ratio: {harmonic_ratio}\n"
        f"Technical interpretation: {interpretation}"
    )


def generate_all_text_summaries(df: pd.DataFrame | None = None) -> list[str]:
    """Generate readable text summaries for all scenarios."""
    summaries_df = generate_scenario_summaries(df)

    return [
        generate_text_summary(row)
        for _, row in summaries_df.iterrows()
    ]


def main() -> None:
    """Run a quick manual validation."""
    summaries_df = generate_scenario_summaries()

    print("Scenario summaries")
    print(summaries_df)

    print("\nText summaries")
    for text_summary in generate_all_text_summaries():
        print("\n" + "-" * 80)
        print(text_summary)


if __name__ == "__main__":
    main()