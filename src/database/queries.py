"""
Database query utilities for the Hybrid AI Predictive Maintenance project.

This module centralizes read operations from the SQLite database.
It will be used by the Streamlit app, ML layer, Text-to-SQL module,
agents, and human-in-the-loop validation flow.
"""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "hybrid_ai_predictive_maintenance.sqlite3"


def get_connection() -> sqlite3.Connection:
    """Create a SQLite database connection."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DATABASE_PATH}. "
            "Run src/database/create_database.py and src/database/seed_database.py first."
        )

    return sqlite3.connect(DATABASE_PATH)


def read_sql_query(query: str, params: tuple | None = None) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with get_connection() as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_assets() -> pd.DataFrame:
    """Return all registered industrial assets."""
    query = """
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

    return read_sql_query(query)


def get_scenarios() -> pd.DataFrame:
    """Return all available maintenance scenarios."""
    query = """
    SELECT
        scenario_id,
        scenario_name,
        scenario_label,
        description,
        severity_level
    FROM scenarios
    ORDER BY scenario_id;
    """

    return read_sql_query(query)


def get_measurements() -> pd.DataFrame:
    """Return vibration measurements enriched with asset and scenario information."""
    query = """
    SELECT
        vm.measurement_id,
        vm.timestamp,
        a.asset_id,
        a.asset_name,
        a.asset_type,
        a.location,
        s.scenario_id,
        s.scenario_name,
        s.scenario_label,
        s.severity_level,
        vm.sensor_position,
        vm.sampling_rate_hz,
        vm.signal_duration_seconds,
        vm.rms_velocity,
        vm.peak_velocity,
        vm.crest_factor,
        vm.temperature_celsius,
        vm.load_percentage
    FROM vibration_measurements vm
    JOIN assets a
        ON vm.asset_id = a.asset_id
    JOIN scenarios s
        ON vm.scenario_id = s.scenario_id
    ORDER BY vm.timestamp;
    """

    return read_sql_query(query)


def get_measurements_by_scenario(scenario_name: str) -> pd.DataFrame:
    """Return vibration measurements filtered by scenario name."""
    query = """
    SELECT
        vm.measurement_id,
        vm.timestamp,
        a.asset_name,
        a.asset_type,
        a.location,
        s.scenario_name,
        s.scenario_label,
        s.severity_level,
        vm.sensor_position,
        vm.rms_velocity,
        vm.peak_velocity,
        vm.crest_factor,
        vm.temperature_celsius,
        vm.load_percentage,
        sf.dominant_frequency_hz,
        sf.low_frequency_energy,
        sf.mid_frequency_energy,
        sf.high_frequency_energy,
        sf.broadband_energy,
        sf.harmonic_ratio,
        sf.subharmonic_ratio,
        sf.anomaly_score,
        md.anomaly_probability,
        md.predicted_label,
        md.model_name,
        md.model_version,
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
    WHERE s.scenario_name = ?
    ORDER BY vm.timestamp;
    """

    return read_sql_query(query, params=(scenario_name,))


def get_spectral_features() -> pd.DataFrame:
    """Return spectral features enriched with measurement, asset, and scenario information."""
    query = """
    SELECT
        sf.feature_id,
        sf.measurement_id,
        vm.timestamp,
        a.asset_name,
        s.scenario_name,
        s.scenario_label,
        sf.dominant_frequency_hz,
        sf.low_frequency_energy,
        sf.mid_frequency_energy,
        sf.high_frequency_energy,
        sf.broadband_energy,
        sf.harmonic_ratio,
        sf.subharmonic_ratio,
        sf.anomaly_score
    FROM spectral_features sf
    JOIN vibration_measurements vm
        ON sf.measurement_id = vm.measurement_id
    JOIN assets a
        ON vm.asset_id = a.asset_id
    JOIN scenarios s
        ON vm.scenario_id = s.scenario_id
    ORDER BY sf.measurement_id;
    """

    return read_sql_query(query)


def get_diagnostics() -> pd.DataFrame:
    """Return ML diagnostics enriched with asset and scenario context."""
    query = """
    SELECT
        md.diagnostic_id,
        md.measurement_id,
        vm.timestamp,
        a.asset_name,
        s.scenario_name,
        s.scenario_label,
        md.predicted_label,
        md.anomaly_probability,
        md.model_name,
        md.model_version,
        md.explanation
    FROM ml_diagnostics md
    JOIN vibration_measurements vm
        ON md.measurement_id = vm.measurement_id
    JOIN assets a
        ON vm.asset_id = a.asset_id
    JOIN scenarios s
        ON vm.scenario_id = s.scenario_id
    ORDER BY md.measurement_id;
    """

    return read_sql_query(query)


def get_full_diagnostic_view() -> pd.DataFrame:
    """Return a consolidated view with measurements, spectral features, and diagnostics."""
    query = """
    SELECT
        vm.measurement_id,
        vm.timestamp,
        a.asset_name,
        a.asset_type,
        a.location,
        s.scenario_name,
        s.scenario_label,
        s.severity_level,
        vm.sensor_position,
        vm.rms_velocity,
        vm.peak_velocity,
        vm.crest_factor,
        vm.temperature_celsius,
        vm.load_percentage,
        sf.dominant_frequency_hz,
        sf.low_frequency_energy,
        sf.mid_frequency_energy,
        sf.high_frequency_energy,
        sf.broadband_energy,
        sf.harmonic_ratio,
        sf.subharmonic_ratio,
        sf.anomaly_score,
        md.diagnostic_id,
        md.predicted_label,
        md.anomaly_probability,
        md.model_name,
        md.model_version,
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
    ORDER BY vm.timestamp;
    """

    return read_sql_query(query)


def get_scenario_by_id(scenario_id: int) -> dict | None:
    """
    Return one maintenance scenario filtered by scenario_id.
    """

    scenarios_df = get_scenarios()

    filtered_df = scenarios_df[scenarios_df["scenario_id"] == scenario_id]

    if filtered_df.empty:
        return None

    return filtered_df.iloc[0].to_dict()


def main() -> None:
    """Quick validation for manual execution."""
    print("Assets")
    print(get_assets())

    print("\nScenarios")
    print(get_scenarios())

    print("\nFull diagnostic view")
    print(get_full_diagnostic_view())


if __name__ == "__main__":
    main()