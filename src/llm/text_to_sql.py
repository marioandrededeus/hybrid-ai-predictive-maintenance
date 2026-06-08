"""
Mock Text-to-SQL module for the Hybrid AI Predictive Maintenance project.

This module simulates a safe Text-to-SQL flow before connecting a real LLM.
It maps natural language questions to controlled SQL SELECT queries,
validates them with sql_guard, and executes them against SQLite.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import read_sql_query  # noqa: E402
from src.llm.sql_guard import validate_sql_query  # noqa: E402


def map_question_to_sql(question: str) -> str:
    """
    Map a natural language question to a controlled SQL query.

    This is intentionally simple and deterministic.
    Later, this function can be replaced or extended with an LLM.
    """
    normalized_question = question.lower().strip()

    if any(term in normalized_question for term in ["scenario", "scenarios", "cenário", "cenarios"]):
        return """
        SELECT
            scenario_id,
            scenario_name,
            scenario_label,
            description,
            severity_level
        FROM scenarios
        ORDER BY scenario_id;
        """

    if any(term in normalized_question for term in ["asset", "assets", "ativo", "ativos", "equipment"]):
        return """
        SELECT
            asset_id,
            asset_name,
            asset_type,
            location,
            manufacturer,
            installation_year
        FROM assets
        ORDER BY asset_id;
        """

    if any(term in normalized_question for term in ["anomaly", "anomalia", "risk", "risco"]):
        return """
        SELECT
            vm.measurement_id,
            vm.timestamp,
            a.asset_name,
            s.scenario_label,
            d.overall_anomaly_score,
            md.anomaly_probability,
            md.predicted_label
        FROM vibration_measurements vm
        JOIN assets a
            ON vm.asset_id = a.asset_id
        JOIN scenarios s
            ON vm.scenario_id = s.scenario_id
        LEFT JOIN spectral_features sf
            ON vm.measurement_id = sf.measurement_id
        LEFT JOIN ml_diagnostics md
            ON vm.measurement_id = md.measurement_id
        ORDER BY d.overall_anomaly_score DESC;
        """

    if any(term in normalized_question for term in ["lubrication", "lubrificação", "carpet"]):
        return """
        SELECT
            vm.measurement_id,
            vm.timestamp,
            a.asset_name,
            s.scenario_label,
            sf.high_frequency_energy,
            sf.broadband_energy,
            d.overall_anomaly_score,
            md.explanation
        FROM vibration_measurements vm
        JOIN assets a
            ON vm.asset_id = a.asset_id
        JOIN scenarios s
            ON vm.scenario_id = s.scenario_id
        LEFT JOIN spectral_features sf
            ON vm.measurement_id = sf.measurement_id
        LEFT JOIN ml_diagnostics md
            ON vm.measurement_id = md.measurement_id
        WHERE s.scenario_name = 'carpet_lubrication_issue'
        ORDER BY d.overall_anomaly_score DESC;
        """

    if any(term in normalized_question for term in ["looseness", "folga", "structural"]):
        return """
        SELECT
            vm.measurement_id,
            vm.timestamp,
            a.asset_name,
            s.scenario_label,
            sf.low_frequency_energy,
            sf.harmonic_ratio,
            sf.subharmonic_ratio,
            d.overall_anomaly_score,
            md.explanation
        FROM vibration_measurements vm
        JOIN assets a
            ON vm.asset_id = a.asset_id
        JOIN scenarios s
            ON vm.scenario_id = s.scenario_id
        LEFT JOIN spectral_features sf
            ON vm.measurement_id = sf.measurement_id
        LEFT JOIN ml_diagnostics md
            ON vm.measurement_id = md.measurement_id
        WHERE s.scenario_name = 'structural_looseness'
        ORDER BY d.overall_anomaly_score DESC;
        """

    if any(term in normalized_question for term in ["feedback", "validation", "validação", "human"]):
        return """
        SELECT
            hf.feedback_id,
            hf.created_at,
            hf.user_name,
            hf.measurement_id,
            a.asset_name,
            s.scenario_label,
            md.predicted_label,
            hf.validated_label,
            hf.feedback_notes,
            hf.action_required
        FROM human_feedback hf
        JOIN vibration_measurements vm
            ON hf.measurement_id = vm.measurement_id
        JOIN assets a
            ON vm.asset_id = a.asset_id
        JOIN scenarios s
            ON vm.scenario_id = s.scenario_id
        LEFT JOIN ml_diagnostics md
            ON hf.diagnostic_id = md.diagnostic_id
        ORDER BY hf.created_at DESC;
        """

    return """
    SELECT
        vm.measurement_id,
        vm.timestamp,
        a.asset_name,
        a.asset_type,
        s.scenario_label,
        vm.rms_velocity,
        vm.peak_velocity,
        d.overall_anomaly_score,
        md.predicted_label,
        md.anomaly_probability
    FROM vibration_measurements vm
    JOIN assets a
        ON vm.asset_id = a.asset_id
    JOIN scenarios s
        ON vm.scenario_id = s.scenario_id
    LEFT JOIN spectral_features sf
        ON vm.measurement_id = sf.measurement_id
    LEFT JOIN ml_diagnostics md
        ON vm.measurement_id = md.measurement_id
    ORDER BY vm.timestamp;
    """


def run_text_to_sql(question: str) -> tuple[str, pd.DataFrame, str]:
    """
    Convert a question to SQL, validate the SQL, and execute the query.

    Returns:
        tuple[str, pd.DataFrame, str]: generated SQL, result DataFrame, validation message.
    """
    sql_query = map_question_to_sql(question)

    is_valid, validation_message = validate_sql_query(sql_query)

    if not is_valid:
        return sql_query, pd.DataFrame(), validation_message

    result_df = read_sql_query(sql_query)

    return sql_query, result_df, validation_message


def main() -> None:
    """Run manual validation examples."""
    questions = [
        "Show me all scenarios",
        "Which assets are monitored?",
        "Show anomaly risk by measurement",
        "Show lubrication issues",
        "Show structural looseness cases",
        "Show human validation history",
    ]

    for question in questions:
        sql_query, result_df, validation_message = run_text_to_sql(question)

        print("\n" + "=" * 100)
        print(f"Question: {question}")
        print(f"Validation: {validation_message}")
        print("Generated SQL:")
        print(sql_query.strip())
        print("Result:")
        print(result_df)


if __name__ == "__main__":
    main()