"""
Database query helpers for the Hybrid AI Predictive Maintenance project.

This module centralizes all SQLite reads used by:
- Streamlit monitoring screen
- Chat GenAI SQL templates
- diagnostic cards
- raw vibration signal charts

The database now follows this flow:

raw vibration signal
-> spectral features
-> physics-informed condition scores
-> diagnostic view
-> app visualization
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.create_database import DATABASE_PATH


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """
    Create a SQLite connection with foreign keys enabled.
    """

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def read_sql_query(
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Execute a SELECT query and return a pandas DataFrame.

    Parameters
    ----------
    query:
        SQL SELECT query.
    params:
        Optional SQL parameters.
    db_path:
        SQLite database path.
    """

    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_scenarios(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """
    Return all diagnostic scenarios.
    """

    query = """
        SELECT
            scenario_id,
            scenario_name,
            scenario_label,
            scenario_description,
            expected_fault_pattern,
            created_at
        FROM scenarios
        ORDER BY scenario_id;
    """

    return read_sql_query(query, db_path=db_path)

def get_scenario_by_id(
    scenario_id: int,
    db_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    """
    Return one diagnostic scenario by ID.

    Kept for backward compatibility with the maintenance agent.
    The maintenance agent expects:
    - a dictionary when the scenario exists
    - None when the scenario does not exist
    """

    query = """
        SELECT
            scenario_id,
            scenario_name,
            scenario_label,
            scenario_description,
            expected_fault_pattern,
            created_at
        FROM scenarios
        WHERE scenario_id = ?;
    """

    df = read_sql_query(query, params=(scenario_id,), db_path=db_path)

    if df.empty:
        return None

    return df.iloc[0].to_dict()

def get_assets(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """
    Return all active assets.
    """

    query = """
        SELECT
            asset_id,
            asset_code,
            asset_name,
            asset_type,
            plant_area,
            manufacturer,
            criticality,
            is_active,
            created_at
        FROM assets
        WHERE is_active = 1
        ORDER BY asset_code;
    """

    return read_sql_query(query, db_path=db_path)


def get_available_collection_dates(
    db_path: Path = DATABASE_PATH,
) -> list[str]:
    """
    Return all available collection dates.

    Used by the Streamlit sidebar date selector.
    """

    query = """
        SELECT DISTINCT
            collection_date
        FROM vibration_measurements
        ORDER BY collection_date;
    """

    df = read_sql_query(query, db_path=db_path)

    if df.empty:
        return []

    return df["collection_date"].tolist()


def get_min_max_collection_dates(
    db_path: Path = DATABASE_PATH,
) -> tuple[str | None, str | None]:
    """
    Return the minimum and maximum collection dates available in the database.
    """

    query = """
        SELECT
            MIN(collection_date) AS min_collection_date,
            MAX(collection_date) AS max_collection_date
        FROM vibration_measurements;
    """

    df = read_sql_query(query, db_path=db_path)

    if df.empty:
        return None, None

    return (
        df.loc[0, "min_collection_date"],
        df.loc[0, "max_collection_date"],
    )


def get_full_diagnostic_view(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return the full diagnostic view.

    This joins:
    - assets
    - scenarios
    - vibration measurements
    - spectral features
    - calculated diagnostics

    Compatibility columns:
    - location maps to plant_area
    - severity_level maps new severity to legacy values
    - anomaly_score maps to overall_anomaly_score
    """

    query = """
        SELECT
            m.measurement_id,
            m.asset_id,
            a.asset_code,
            a.asset_name,
            a.asset_type,
            a.plant_area AS location,
            a.plant_area,
            a.manufacturer,
            a.criticality,

            m.scenario_id,
            s.scenario_name,
            s.scenario_label,
            s.scenario_description,
            s.expected_fault_pattern,

            m.timestamp,
            m.collection_date,
            m.sensor_position,
            m.axis,
            m.sampling_rate_hz,
            m.duration_seconds,
            m.rpm,

            m.rms_velocity_mm_s,
            m.rms_velocity_mm_s AS rms_velocity,
            m.peak_velocity_mm_s,
            m.peak_velocity_mm_s AS peak_velocity,
            m.crest_factor,
            m.temperature_c,
            m.operating_condition,
            m.notes,

            sf.rotational_frequency_hz,
            sf.low_frequency_energy,
            sf.mid_frequency_energy,
            sf.high_frequency_energy,
            sf.broadband_energy,
            sf.spectral_floor_level,
            sf.spectral_flatness,
            sf.high_frequency_ratio,
            sf.amplitude_0_5x,
            sf.amplitude_1x,
            sf.amplitude_1_5x,
            sf.amplitude_2x,
            sf.amplitude_3x,
            sf.harmonic_ratio,
            sf.subharmonic_ratio,
            sf.dominant_frequency_hz,
            sf.dominant_amplitude,
            sf.spectral_centroid_hz,
            sf.spectral_kurtosis,

            d.carpet_score,
            d.looseness_score,
            d.overall_anomaly_score,
            d.overall_anomaly_score AS anomaly_score,
            d.anomaly_probability,
            d.predicted_condition,
            d.severity,

            CASE
                WHEN d.severity = 'Critical' THEN 'High'
                WHEN d.severity = 'Attention' THEN 'Medium'
                ELSE 'Low'
            END AS severity_level,

            d.diagnostic_method,
            d.model_name,
            d.model_version,
            d.diagnostic_explanation,
            d.recommended_action

        FROM vibration_measurements AS m
        INNER JOIN assets AS a
            ON m.asset_id = a.asset_id
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        LEFT JOIN spectral_features AS sf
            ON m.measurement_id = sf.measurement_id
        LEFT JOIN ml_diagnostics AS d
            ON m.measurement_id = d.measurement_id
        ORDER BY
            m.timestamp DESC,
            a.asset_code ASC;
    """

    return read_sql_query(query, db_path=db_path)


def get_latest_measurements_until_date(
    selected_date: str,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return the latest available measurement of each asset up to a selected date.

    This supports the monitoring screen:

    user selects date
    -> app finds latest reading per equipment up to that date
    -> app renders one card per equipment
    """

    query = """
        WITH ranked_measurements AS (
            SELECT
                m.measurement_id,
                ROW_NUMBER() OVER (
                    PARTITION BY m.asset_id
                    ORDER BY m.timestamp DESC
                ) AS row_number
            FROM vibration_measurements AS m
            WHERE m.collection_date <= ?
        )

        SELECT
            full_view.*
        FROM ranked_measurements AS ranked
        INNER JOIN (
            SELECT
                m.measurement_id,
                m.asset_id,
                a.asset_code,
                a.asset_name,
                a.asset_type,
                a.plant_area,
                a.manufacturer,
                a.criticality,

                m.scenario_id,
                s.scenario_name,
                s.scenario_label,
                s.scenario_description,
                s.expected_fault_pattern,

                m.timestamp,
                m.collection_date,
                m.sensor_position,
                m.axis,
                m.sampling_rate_hz,
                m.duration_seconds,
                m.rpm,

                m.rms_velocity_mm_s,
                m.peak_velocity_mm_s,
                m.crest_factor,
                m.temperature_c,
                m.operating_condition,
                m.notes,

                sf.rotational_frequency_hz,
                sf.low_frequency_energy,
                sf.mid_frequency_energy,
                sf.high_frequency_energy,
                sf.broadband_energy,
                sf.spectral_floor_level,
                sf.spectral_flatness,
                sf.high_frequency_ratio,
                sf.amplitude_0_5x,
                sf.amplitude_1x,
                sf.amplitude_1_5x,
                sf.amplitude_2x,
                sf.amplitude_3x,
                sf.harmonic_ratio,
                sf.subharmonic_ratio,
                sf.dominant_frequency_hz,
                sf.dominant_amplitude,
                sf.spectral_centroid_hz,
                sf.spectral_kurtosis,

                d.carpet_score,
                d.looseness_score,
                d.overall_anomaly_score,
                d.anomaly_probability,
                d.predicted_condition,
                d.severity,
                d.diagnostic_method,
                d.model_name,
                d.model_version,
                d.diagnostic_explanation,
                d.recommended_action

            FROM vibration_measurements AS m
            INNER JOIN assets AS a
                ON m.asset_id = a.asset_id
            INNER JOIN scenarios AS s
                ON m.scenario_id = s.scenario_id
            LEFT JOIN spectral_features AS sf
                ON m.measurement_id = sf.measurement_id
            LEFT JOIN ml_diagnostics AS d
                ON m.measurement_id = d.measurement_id
        ) AS full_view
            ON ranked.measurement_id = full_view.measurement_id
        WHERE ranked.row_number = 1
        ORDER BY
            full_view.severity DESC,
            full_view.asset_code ASC;
    """

    return read_sql_query(query, params=(selected_date,), db_path=db_path)


def get_latest_measurements(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """
    Return the latest measurement of each asset considering the full database.

    Kept for backward compatibility with previous app versions.
    """

    _, max_collection_date = get_min_max_collection_dates(db_path=db_path)

    if max_collection_date is None:
        return pd.DataFrame()

    return get_latest_measurements_until_date(
        selected_date=max_collection_date,
        db_path=db_path,
    )


def get_measurements_by_scenario(
    scenario: int | str,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return full diagnostic rows for a specific scenario.

    Accepts either:
    - scenario_id as int
    - scenario_name as str

    This keeps compatibility with the maintenance agent, which calls this
    function using scenario["scenario_name"].
    """

    base_query = """
        SELECT
            m.measurement_id,
            m.asset_id,
            a.asset_code,
            a.asset_name,
            a.asset_type,
            a.plant_area AS location,
            a.plant_area,
            a.manufacturer,
            a.criticality,

            m.scenario_id,
            s.scenario_name,
            s.scenario_label,

            m.timestamp,
            m.collection_date,
            m.sampling_rate_hz,
            m.duration_seconds,
            m.rpm,

            m.rms_velocity_mm_s,
            m.peak_velocity_mm_s,
            m.crest_factor,
            m.temperature_c,

            sf.low_frequency_energy,
            sf.mid_frequency_energy,
            sf.high_frequency_energy,
            sf.broadband_energy,
            sf.spectral_floor_level,
            sf.spectral_flatness,
            sf.high_frequency_ratio,
            sf.harmonic_ratio,
            sf.subharmonic_ratio,

            d.carpet_score,
            d.looseness_score,
            d.overall_anomaly_score,
            d.overall_anomaly_score AS anomaly_score,
            d.anomaly_probability,
            d.predicted_condition,
            d.severity,
            d.severity AS severity_level,
            d.diagnostic_explanation,
            d.recommended_action

        FROM vibration_measurements AS m
        INNER JOIN assets AS a
            ON m.asset_id = a.asset_id
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        LEFT JOIN spectral_features AS sf
            ON m.measurement_id = sf.measurement_id
        LEFT JOIN ml_diagnostics AS d
            ON m.measurement_id = d.measurement_id
    """

    if isinstance(scenario, int):
        query = base_query + """
            WHERE m.scenario_id = ?
            ORDER BY m.timestamp DESC;
        """
        params = (scenario,)
    else:
        query = base_query + """
            WHERE s.scenario_name = ?
            ORDER BY m.timestamp DESC;
        """
        params = (scenario,)

    return read_sql_query(query, params=params, db_path=db_path)


def get_raw_vibration_samples(
    measurement_id: int,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return raw vibration samples for a measurement.

    Used by the app to plot:
    - waveform
    - FFT
    - frequency bands
    - harmonic markers
    """

    query = """
        SELECT
            sample_index,
            time_seconds,
            acceleration_g,
            velocity_mm_s
        FROM vibration_raw_samples
        WHERE measurement_id = ?
        ORDER BY sample_index;
    """

    return read_sql_query(query, params=(measurement_id,), db_path=db_path)


def get_feature_diagnostics(
    measurement_id: int,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return feature-level diagnostics for a measurement.
    """

    query = """
        SELECT
            feature_diagnostic_id,
            measurement_id,
            feature_name,
            feature_value,
            reference_value,
            diagnostic_flag,
            diagnostic_reason,
            created_at
        FROM feature_diagnostics
        WHERE measurement_id = ?
        ORDER BY feature_diagnostic_id;
    """

    return read_sql_query(query, params=(measurement_id,), db_path=db_path)


def get_high_severity_diagnostics(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return high severity diagnostics.

    Compatibility note:
    Older tests expect severity_level = 'High'.

    The new schema uses severity = 'Critical' / 'Attention' / 'Normal'.
    For backward compatibility, this function maps Critical diagnostics to
    severity_level = 'High' while preserving the new severity column.
    """

    query = """
        SELECT
            m.measurement_id,
            m.timestamp,

            a.asset_code,
            a.asset_name,
            a.asset_type,
            a.plant_area AS location,
            a.plant_area,
            a.criticality,

            s.scenario_name,
            s.scenario_label,

            'High' AS severity_level,
            d.severity,

            d.overall_anomaly_score AS anomaly_score,
            d.overall_anomaly_score,
            d.carpet_score,
            d.looseness_score,
            d.anomaly_probability,
            d.predicted_condition,
            d.recommended_action

        FROM ml_diagnostics AS d
        INNER JOIN vibration_measurements AS m
            ON d.measurement_id = m.measurement_id
        INNER JOIN assets AS a
            ON m.asset_id = a.asset_id
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        WHERE d.overall_anomaly_score >= 0.45
        ORDER BY
            d.overall_anomaly_score DESC,
            m.timestamp DESC;
    """

    return read_sql_query(query, db_path=db_path)


def get_average_anomaly_score_by_scenario(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return average anomaly score grouped by scenario.
    """

    query = """
        SELECT
            s.scenario_name,
            s.scenario_label,
            ROUND(AVG(d.overall_anomaly_score), 4) AS average_anomaly_score,
            ROUND(AVG(d.carpet_score), 4) AS average_carpet_score,
            ROUND(AVG(d.looseness_score), 4) AS average_looseness_score,
            COUNT(*) AS measurements_count
        FROM ml_diagnostics AS d
        INNER JOIN vibration_measurements AS m
            ON d.measurement_id = m.measurement_id
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        GROUP BY
            s.scenario_name,
            s.scenario_label
        ORDER BY average_anomaly_score DESC;
    """

    return read_sql_query(query, db_path=db_path)


def get_average_anomaly_probability_by_scenario(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return average anomaly probability grouped by scenario.
    """

    query = """
        SELECT
            s.scenario_name,
            s.scenario_label,
            ROUND(AVG(d.anomaly_probability), 4) AS average_anomaly_probability,
            COUNT(*) AS measurements_count
        FROM ml_diagnostics AS d
        INNER JOIN vibration_measurements AS m
            ON d.measurement_id = m.measurement_id
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        GROUP BY
            s.scenario_name,
            s.scenario_label
        ORDER BY average_anomaly_probability DESC;
    """

    return read_sql_query(query, db_path=db_path)


def get_average_rms_velocity_by_scenario(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return average RMS velocity grouped by scenario.
    """

    query = """
        SELECT
            s.scenario_name,
            s.scenario_label,
            ROUND(AVG(m.rms_velocity_mm_s), 4) AS average_rms_velocity_mm_s,
            COUNT(*) AS measurements_count
        FROM vibration_measurements AS m
        INNER JOIN scenarios AS s
            ON m.scenario_id = s.scenario_id
        GROUP BY
            s.scenario_name,
            s.scenario_label
        ORDER BY average_rms_velocity_mm_s DESC;
    """

    return read_sql_query(query, db_path=db_path)


def get_assets_with_high_risk(
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return assets with high-risk diagnostics.

    Kept for current Chat GenAI templates.
    """

    query = """
        SELECT
            a.asset_code,
            a.asset_name,
            a.asset_type,
            a.plant_area,
            a.criticality,
            COUNT(*) AS critical_events,
            ROUND(MAX(d.overall_anomaly_score), 4) AS max_anomaly_score,
            ROUND(MAX(d.anomaly_probability), 4) AS max_anomaly_probability
        FROM ml_diagnostics AS d
        INNER JOIN vibration_measurements AS m
            ON d.measurement_id = m.measurement_id
        INNER JOIN assets AS a
            ON m.asset_id = a.asset_id
        WHERE d.severity = 'Critical'
        GROUP BY
            a.asset_code,
            a.asset_name,
            a.asset_type,
            a.plant_area,
            a.criticality
        ORDER BY
            critical_events DESC,
            max_anomaly_score DESC;
    """

    return read_sql_query(query, db_path=db_path)


def get_condition_trend_by_asset(
    asset_id: int,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    Return condition trend for one asset.

    Useful for future trend charts.
    """

    query = """
        SELECT
            m.measurement_id,
            m.asset_id,
            a.asset_code,
            a.asset_name,
            m.timestamp,
            m.collection_date,
            m.rms_velocity_mm_s,
            d.carpet_score,
            d.looseness_score,
            d.overall_anomaly_score,
            d.anomaly_probability,
            d.predicted_condition,
            d.severity
        FROM vibration_measurements AS m
        INNER JOIN assets AS a
            ON m.asset_id = a.asset_id
        LEFT JOIN ml_diagnostics AS d
            ON m.measurement_id = d.measurement_id
        WHERE m.asset_id = ?
        ORDER BY m.timestamp;
    """

    return read_sql_query(query, params=(asset_id,), db_path=db_path)


def get_database_summary(
    db_path: Path = DATABASE_PATH,
) -> dict[str, int]:
    """
    Return simple database counts for diagnostics and tests.
    """

    with get_connection(db_path) as connection:
        measurements_count = connection.execute(
            "SELECT COUNT(*) FROM vibration_measurements;"
        ).fetchone()[0]

        raw_samples_count = connection.execute(
            "SELECT COUNT(*) FROM vibration_raw_samples;"
        ).fetchone()[0]

        diagnostics_count = connection.execute(
            "SELECT COUNT(*) FROM ml_diagnostics;"
        ).fetchone()[0]

        assets_count = connection.execute(
            "SELECT COUNT(*) FROM assets;"
        ).fetchone()[0]

    return {
        "assets_count": int(assets_count),
        "measurements_count": int(measurements_count),
        "raw_samples_count": int(raw_samples_count),
        "diagnostics_count": int(diagnostics_count),
    }