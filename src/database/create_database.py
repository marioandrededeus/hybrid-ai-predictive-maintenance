"""
Create SQLite database schema for the Hybrid AI Predictive Maintenance project.

This schema supports a physics-informed diagnostic flow:

raw vibration signal
-> signal processing
-> spectral features
-> carpet / looseness scores
-> condition classification
-> rule-based action plan
-> future GenAI / Agent / Human-in-the-loop layers

Important:
The database stores raw vibration samples and calculated outputs.
Diagnostics must be generated from the raw signal processing pipeline,
not manually created as ready-made labels.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "hybrid_ai_predictive_maintenance.sqlite3"


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """
    Create a SQLite connection with foreign keys enabled.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """
    Create all database tables.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_name TEXT NOT NULL UNIQUE,
            scenario_label TEXT NOT NULL,
            scenario_description TEXT,
            expected_fault_pattern TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_code TEXT NOT NULL UNIQUE,
            asset_name TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            plant_area TEXT,
            manufacturer TEXT,
            criticality TEXT CHECK (
                criticality IN ('Low', 'Medium', 'High')
            ) DEFAULT 'Medium',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vibration_measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER NOT NULL,
            scenario_id INTEGER NOT NULL,

            timestamp TEXT NOT NULL,
            collection_date TEXT NOT NULL,

            sensor_position TEXT,
            axis TEXT DEFAULT 'radial',

            sampling_rate_hz REAL NOT NULL,
            duration_seconds REAL NOT NULL,
            rpm REAL NOT NULL,

            rms_velocity_mm_s REAL,
            peak_velocity_mm_s REAL,
            crest_factor REAL,
            temperature_c REAL,

            operating_condition TEXT,
            notes TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (asset_id)
                REFERENCES assets(asset_id)
                ON DELETE CASCADE,

            FOREIGN KEY (scenario_id)
                REFERENCES scenarios(scenario_id)
                ON DELETE RESTRICT
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vibration_raw_samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,

            sample_index INTEGER NOT NULL,
            time_seconds REAL NOT NULL,

            acceleration_g REAL,
            velocity_mm_s REAL NOT NULL,

            FOREIGN KEY (measurement_id)
                REFERENCES vibration_measurements(measurement_id)
                ON DELETE CASCADE,

            UNIQUE (measurement_id, sample_index)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS spectral_features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,

            rotational_frequency_hz REAL NOT NULL,

            low_frequency_energy REAL NOT NULL,
            mid_frequency_energy REAL NOT NULL,
            high_frequency_energy REAL NOT NULL,
            broadband_energy REAL NOT NULL,

            spectral_floor_level REAL,
            spectral_flatness REAL,
            high_frequency_ratio REAL,

            amplitude_0_5x REAL,
            amplitude_1x REAL,
            amplitude_1_5x REAL,
            amplitude_2x REAL,
            amplitude_3x REAL,

            harmonic_ratio REAL NOT NULL,
            subharmonic_ratio REAL NOT NULL,

            dominant_frequency_hz REAL,
            dominant_amplitude REAL,

            spectral_centroid_hz REAL,
            spectral_kurtosis REAL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (measurement_id)
                REFERENCES vibration_measurements(measurement_id)
                ON DELETE CASCADE,

            UNIQUE (measurement_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_diagnostics (
            diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,

            carpet_score REAL NOT NULL,
            looseness_score REAL NOT NULL,
            overall_anomaly_score REAL NOT NULL,
            anomaly_probability REAL NOT NULL,

            predicted_condition TEXT CHECK (
                predicted_condition IN (
                    'normal_operation',
                    'carpet_lubrication_issue',
                    'structural_looseness',
                    'mixed_fault_pattern'
                )
            ) NOT NULL,

            severity TEXT CHECK (
                severity IN ('Normal', 'Attention', 'Critical')
            ) NOT NULL,

            diagnostic_method TEXT DEFAULT 'physics_informed_rule_based',
            model_name TEXT DEFAULT 'physics_informed_scoring',
            model_version TEXT DEFAULT 'v1',

            diagnostic_explanation TEXT,
            recommended_action TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (measurement_id)
                REFERENCES vibration_measurements(measurement_id)
                ON DELETE CASCADE,

            UNIQUE (measurement_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_diagnostics (
            feature_diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,

            feature_name TEXT NOT NULL,
            feature_value REAL NOT NULL,
            reference_value REAL,

            diagnostic_flag TEXT CHECK (
                diagnostic_flag IN ('Normal', 'Attention', 'Critical')
            ) NOT NULL,

            diagnostic_reason TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (measurement_id)
                REFERENCES vibration_measurements(measurement_id)
                ON DELETE CASCADE
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS human_validations (
            validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,

            validation_status TEXT CHECK (
                validation_status IN (
                    'Pending',
                    'Confirmed',
                    'Rejected',
                    'Needs Review'
                )
            ) DEFAULT 'Pending',

            validated_by TEXT,
            validation_comment TEXT,
            validated_at TEXT,

            FOREIGN KEY (measurement_id)
                REFERENCES vibration_measurements(measurement_id)
                ON DELETE CASCADE
        );
        """
    )

    connection.commit()


def create_indexes(connection: sqlite3.Connection) -> None:
    """
    Create database indexes used by the app, SQL templates and diagnostics.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurements_asset_timestamp
        ON vibration_measurements (asset_id, timestamp);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurements_collection_date
        ON vibration_measurements (collection_date);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurements_scenario
        ON vibration_measurements (scenario_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_raw_samples_measurement
        ON vibration_raw_samples (measurement_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_raw_samples_measurement_index
        ON vibration_raw_samples (measurement_id, sample_index);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_spectral_features_measurement
        ON spectral_features (measurement_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_diagnostics_measurement
        ON ml_diagnostics (measurement_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_diagnostics_condition
        ON ml_diagnostics (predicted_condition);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_diagnostics_severity
        ON ml_diagnostics (severity);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_diagnostics_measurement
        ON feature_diagnostics (measurement_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_human_validations_measurement
        ON human_validations (measurement_id);
        """
    )

    connection.commit()


def create_database(db_path: Path = DATABASE_PATH) -> None:
    """
    Create database schema without deleting an existing database.
    """

    with get_connection(db_path) as connection:
        create_tables(connection)
        create_indexes(connection)


def reset_database(db_path: Path = DATABASE_PATH) -> None:
    """
    Delete and recreate the database.

    This is recommended during the current demo phase because the schema is still
    evolving and the database will be regenerated by seed_database.py.
    """

    if db_path.exists():
        db_path.unlink()

    create_database(db_path)


if __name__ == "__main__":
    reset_database()
    print(f"Database created successfully at: {DATABASE_PATH}")