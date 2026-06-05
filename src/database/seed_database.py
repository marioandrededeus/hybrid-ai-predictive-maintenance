"""
Seed the SQLite database with initial industrial assets, scenarios,
vibration measurements, spectral features, and ML diagnostics.

The database is intentionally pre-populated.
The Streamlit app will later simulate scenario exploration by reading from SQLite.
"""

from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "hybrid_ai_predictive_maintenance.sqlite3"


def create_connection() -> sqlite3.Connection:
    """Create a SQLite database connection."""
    return sqlite3.connect(DATABASE_PATH)


def clear_tables(connection: sqlite3.Connection) -> None:
    """Clear tables to make the seed process reproducible."""
    cursor = connection.cursor()

    cursor.execute("DELETE FROM action_logs;")
    cursor.execute("DELETE FROM human_feedback;")
    cursor.execute("DELETE FROM ml_diagnostics;")
    cursor.execute("DELETE FROM spectral_features;")
    cursor.execute("DELETE FROM vibration_measurements;")
    cursor.execute("DELETE FROM scenarios;")
    cursor.execute("DELETE FROM assets;")

    connection.commit()


def seed_assets(connection: sqlite3.Connection) -> None:
    """Insert initial industrial assets."""
    cursor = connection.cursor()

    assets = [
        (
            "Pump P-101",
            "Centrifugal pump",
            "Utility area",
            "Generic Pumps Co.",
            2018,
        ),
        (
            "Motor M-202",
            "Electric motor",
            "Production line A",
            "Generic Motors Inc.",
            2020,
        ),
        (
            "Fan F-303",
            "Industrial fan",
            "Ventilation system",
            "Generic Fans Ltd.",
            2019,
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO assets (
            asset_name,
            asset_type,
            location,
            manufacturer,
            installation_year
        )
        VALUES (?, ?, ?, ?, ?);
        """,
        assets,
    )

    connection.commit()


def seed_scenarios(connection: sqlite3.Connection) -> None:
    """Insert initial maintenance scenarios."""
    cursor = connection.cursor()

    scenarios = [
        (
            "normal_operation",
            "Normal operation",
            "Stable vibration behavior with low anomaly score and balanced spectral energy.",
            "low",
        ),
        (
            "carpet_lubrication_issue",
            "Carpet pattern associated with possible lubrication issues",
            "Broadband spectral elevation, especially at higher frequencies, suggesting possible lubrication degradation.",
            "medium",
        ),
        (
            "structural_looseness",
            "Structural looseness",
            "Increased low-frequency energy, harmonic components, and unstable vibration behavior associated with looseness.",
            "high",
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO scenarios (
            scenario_name,
            scenario_label,
            description,
            severity_level
        )
        VALUES (?, ?, ?, ?);
        """,
        scenarios,
    )

    connection.commit()


def seed_measurements(connection: sqlite3.Connection) -> None:
    """Insert sample vibration measurements."""
    cursor = connection.cursor()

    measurements = [
        # asset_id, scenario_id, timestamp, sensor_position, sampling_rate_hz,
        # signal_duration_seconds, rms_velocity, peak_velocity, crest_factor,
        # temperature_celsius, load_percentage
        (
            1,
            1,
            "2026-01-15 08:00:00",
            "Drive end bearing",
            12000,
            10.0,
            1.8,
            4.2,
            2.3,
            62.0,
            74.0,
        ),
        (
            1,
            2,
            "2026-01-16 08:00:00",
            "Drive end bearing",
            12000,
            10.0,
            3.9,
            10.8,
            2.8,
            78.0,
            76.0,
        ),
        (
            2,
            3,
            "2026-01-17 08:00:00",
            "Non-drive end bearing",
            12000,
            10.0,
            5.4,
            18.5,
            3.4,
            71.0,
            82.0,
        ),
        (
            3,
            1,
            "2026-01-18 08:00:00",
            "Fan housing",
            12000,
            10.0,
            2.1,
            5.0,
            2.4,
            58.0,
            69.0,
        ),
        (
            3,
            2,
            "2026-01-19 08:00:00",
            "Fan housing",
            12000,
            10.0,
            4.2,
            12.4,
            3.0,
            75.0,
            72.0,
        ),
        (
            2,
            3,
            "2026-01-20 08:00:00",
            "Motor housing",
            12000,
            10.0,
            6.1,
            21.2,
            3.5,
            73.0,
            85.0,
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO vibration_measurements (
            asset_id,
            scenario_id,
            timestamp,
            sensor_position,
            sampling_rate_hz,
            signal_duration_seconds,
            rms_velocity,
            peak_velocity,
            crest_factor,
            temperature_celsius,
            load_percentage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        measurements,
    )

    connection.commit()


def seed_spectral_features(connection: sqlite3.Connection) -> None:
    """Insert spectral features for each measurement."""
    cursor = connection.cursor()

    features = [
        # measurement_id, dominant_frequency_hz, low_frequency_energy,
        # mid_frequency_energy, high_frequency_energy, broadband_energy,
        # harmonic_ratio, subharmonic_ratio, anomaly_score
        (
            1,
            60.0,
            0.18,
            0.22,
            0.12,
            0.16,
            0.10,
            0.04,
            0.08,
        ),
        (
            2,
            1800.0,
            0.28,
            0.51,
            0.86,
            0.91,
            0.18,
            0.07,
            0.72,
        ),
        (
            3,
            30.0,
            0.89,
            0.56,
            0.34,
            0.61,
            0.74,
            0.48,
            0.88,
        ),
        (
            4,
            55.0,
            0.20,
            0.25,
            0.15,
            0.18,
            0.12,
            0.05,
            0.11,
        ),
        (
            5,
            2200.0,
            0.31,
            0.57,
            0.89,
            0.94,
            0.21,
            0.09,
            0.76,
        ),
        (
            6,
            35.0,
            0.92,
            0.63,
            0.39,
            0.67,
            0.79,
            0.51,
            0.91,
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO spectral_features (
            measurement_id,
            dominant_frequency_hz,
            low_frequency_energy,
            mid_frequency_energy,
            high_frequency_energy,
            broadband_energy,
            harmonic_ratio,
            subharmonic_ratio,
            anomaly_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        features,
    )

    connection.commit()


def seed_ml_diagnostics(connection: sqlite3.Connection) -> None:
    """Insert initial simulated ML diagnostics."""
    cursor = connection.cursor()

    diagnostics = [
        (
            1,
            "normal_operation",
            0.08,
            "baseline_anomaly_classifier",
            "0.1.0",
            "Low anomaly score and balanced spectral energy indicate stable operation.",
        ),
        (
            2,
            "carpet_lubrication_issue",
            0.72,
            "baseline_anomaly_classifier",
            "0.1.0",
            "High broadband and high-frequency energy suggest a carpet pattern associated with possible lubrication issues.",
        ),
        (
            3,
            "structural_looseness",
            0.88,
            "baseline_anomaly_classifier",
            "0.1.0",
            "High low-frequency energy, harmonic ratio, and subharmonic ratio suggest possible structural looseness.",
        ),
        (
            4,
            "normal_operation",
            0.11,
            "baseline_anomaly_classifier",
            "0.1.0",
            "Vibration indicators remain within normal operating behavior.",
        ),
        (
            5,
            "carpet_lubrication_issue",
            0.76,
            "baseline_anomaly_classifier",
            "0.1.0",
            "Broadband spectral elevation is consistent with possible lubrication degradation.",
        ),
        (
            6,
            "structural_looseness",
            0.91,
            "baseline_anomaly_classifier",
            "0.1.0",
            "Low-frequency dominance and harmonic behavior indicate possible structural looseness.",
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO ml_diagnostics (
            measurement_id,
            predicted_label,
            anomaly_probability,
            model_name,
            model_version,
            explanation
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        diagnostics,
    )

    connection.commit()


def main() -> None:
    """Run database seed process."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DATABASE_PATH}. "
            "Run src/database/create_database.py first."
        )

    connection = create_connection()

    try:
        clear_tables(connection)
        seed_assets(connection)
        seed_scenarios(connection)
        seed_measurements(connection)
        seed_spectral_features(connection)
        seed_ml_diagnostics(connection)

        print("Database seeded successfully.")
        print(f"Database path: {DATABASE_PATH}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()