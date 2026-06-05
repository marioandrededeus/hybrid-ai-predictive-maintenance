"""
Human feedback module for the Hybrid AI Predictive Maintenance project.

This module stores human validation records in the SQLite database.
It supports the human-in-the-loop layer of the architecture.
"""

from pathlib import Path
import sqlite3
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import DATABASE_PATH, read_sql_query  # noqa: E402


def get_connection() -> sqlite3.Connection:
    """Create a SQLite database connection."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DATABASE_PATH}. "
            "Run src/database/create_database.py and src/database/seed_database.py first."
        )

    return sqlite3.connect(DATABASE_PATH)


def save_human_feedback(
    measurement_id: int,
    validated_label: str,
    user_name: str | None = None,
    diagnostic_id: int | None = None,
    feedback_notes: str | None = None,
    action_required: str | None = None,
) -> None:
    """Save a human validation record into the database."""
    query = """
    INSERT INTO human_feedback (
        measurement_id,
        diagnostic_id,
        user_name,
        validated_label,
        feedback_notes,
        action_required
    )
    VALUES (?, ?, ?, ?, ?, ?);
    """

    values = (
        measurement_id,
        diagnostic_id,
        user_name,
        validated_label,
        feedback_notes,
        action_required,
    )

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(query, values)
        connection.commit()


def get_human_feedback() -> pd.DataFrame:
    """Return all human feedback records."""
    query = """
    SELECT
        hf.feedback_id,
        hf.measurement_id,
        hf.diagnostic_id,
        hf.user_name,
        hf.validated_label,
        hf.feedback_notes,
        hf.action_required,
        hf.created_at
    FROM human_feedback hf
    ORDER BY hf.created_at DESC;
    """

    return read_sql_query(query)


def get_human_feedback_with_context() -> pd.DataFrame:
    """Return human feedback enriched with asset, scenario, and ML diagnostic context."""
    query = """
    SELECT
        hf.feedback_id,
        hf.created_at,
        hf.user_name,
        hf.measurement_id,
        a.asset_name,
        a.asset_type,
        s.scenario_label,
        md.predicted_label,
        md.anomaly_probability,
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

    return read_sql_query(query)


def delete_human_feedback(feedback_id: int) -> None:
    """Delete a human feedback record by feedback ID."""
    query = """
    DELETE FROM human_feedback
    WHERE feedback_id = ?;
    """

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(query, (feedback_id,))
        connection.commit()


def main() -> None:
    """Run a quick manual validation."""
    save_human_feedback(
        measurement_id=2,
        diagnostic_id=2,
        user_name="Maintenance Specialist",
        validated_label="carpet_lubrication_issue",
        feedback_notes=(
            "Observed broadband spectral elevation. Lubrication inspection recommended."
        ),
        action_required="Inspect lubrication condition and bearing status.",
    )

    feedback_df = get_human_feedback_with_context()

    print("Human feedback records")
    print(feedback_df)


if __name__ == "__main__":
    main()