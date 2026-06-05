"""
Create and initialize the SQLite database for the Hybrid AI Predictive Maintenance project.

This script creates the first database schema used by the Streamlit app,
ML modules, Text-to-SQL layer, agents, RAG recommendations, and human validation flow.
"""

from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "hybrid_ai_predictive_maintenance.sqlite3"


def create_connection() -> sqlite3.Connection:
    """Create a SQLite database connection."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def create_tables(connection: sqlite3.Connection) -> None:
    """Create the initial database tables."""

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            location TEXT,
            manufacturer TEXT,
            installation_year INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_name TEXT NOT NULL UNIQUE,
            scenario_label TEXT NOT NULL,
            description TEXT,
            severity_level TEXT,
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
            sensor_position TEXT NOT NULL,
            sampling_rate_hz INTEGER NOT NULL,
            signal_duration_seconds REAL NOT NULL,
            rms_velocity REAL,
            peak_velocity REAL,
            crest_factor REAL,
            temperature_celsius REAL,
            load_percentage REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS spectral_features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,
            dominant_frequency_hz REAL,
            low_frequency_energy REAL,
            mid_frequency_energy REAL,
            high_frequency_energy REAL,
            broadband_energy REAL,
            harmonic_ratio REAL,
            subharmonic_ratio REAL,
            anomaly_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (measurement_id) REFERENCES vibration_measurements(measurement_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_diagnostics (
            diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,
            predicted_label TEXT NOT NULL,
            anomaly_probability REAL,
            model_name TEXT,
            model_version TEXT,
            explanation TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (measurement_id) REFERENCES vibration_measurements(measurement_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS human_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,
            diagnostic_id INTEGER,
            user_name TEXT,
            validated_label TEXT NOT NULL,
            feedback_notes TEXT,
            action_required TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (measurement_id) REFERENCES vibration_measurements(measurement_id),
            FOREIGN KEY (diagnostic_id) REFERENCES ml_diagnostics(diagnostic_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS action_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER,
            agent_name TEXT,
            tool_name TEXT,
            user_query TEXT,
            generated_sql TEXT,
            agent_response TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (measurement_id) REFERENCES vibration_measurements(measurement_id)
        );
        """
    )

    connection.commit()


def main() -> None:
    """Run database creation."""
    connection = create_connection()

    try:
        create_tables(connection)
        print(f"Database created successfully at: {DATABASE_PATH}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()