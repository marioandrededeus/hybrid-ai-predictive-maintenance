"""
Seed the Hybrid AI Predictive Maintenance database.

This script creates a synthetic but technically coherent vibration dataset.

Main idea:
raw vibration signal
-> FFT-based feature extraction
-> physics-informed carpet and looseness scores
-> diagnostic classification
-> SQLite persistence

The diagnostic labels are not manually assigned as final truth.
They are calculated from vibration signal features.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.database.create_database import DATABASE_PATH, reset_database


RANDOM_SEED = 42

SAMPLING_RATE_HZ = 1024
DURATION_SECONDS = 2
N_SAMPLES = SAMPLING_RATE_HZ * DURATION_SECONDS

N_DAYS = 30
MEASUREMENTS_PER_DAY = 3

START_DATE = datetime(2026, 5, 1, 8, 0, 0)


SCENARIOS = [
    {
        "scenario_name": "normal_operation",
        "scenario_label": "Normal Operation",
        "scenario_description": "Stable vibration behavior with dominant rotational component and low broadband noise.",
        "expected_fault_pattern": "Dominant 1x rotational component, low noise floor and no relevant harmonic or broadband abnormality.",
    },
    {
        "scenario_name": "carpet_lubrication_issue",
        "scenario_label": "Carpet / Lubrication Issue",
        "scenario_description": "Lubrication-related abnormality represented by increased broadband energy and elevated spectral floor.",
        "expected_fault_pattern": "Broadband spectral floor elevation, high-frequency energy increase and random micro-impacts.",
    },
    {
        "scenario_name": "structural_looseness",
        "scenario_label": "Structural Looseness",
        "scenario_description": "Mechanical looseness pattern represented by low-frequency components, harmonics and sub-harmonics.",
        "expected_fault_pattern": "Increased 0.5x, 1x, 1.5x, 2x and 3x rotational components with higher low-frequency energy.",
    },
]


ASSETS = [
    {
        "asset_code": "PUMP-101",
        "asset_name": "Cooling Water Pump 101",
        "asset_type": "Centrifugal Pump",
        "plant_area": "Utilities",
        "manufacturer": "Demo Industrial",
        "criticality": "High",
        "base_rpm": 1780,
    },
    {
        "asset_code": "FAN-201",
        "asset_name": "Exhaust Fan 201",
        "asset_type": "Industrial Fan",
        "plant_area": "Kiln Area",
        "manufacturer": "Demo Industrial",
        "criticality": "Medium",
        "base_rpm": 1450,
    },
    {
        "asset_code": "MOTOR-301",
        "asset_name": "Main Drive Motor 301",
        "asset_type": "Electric Motor",
        "plant_area": "Production Line",
        "manufacturer": "Demo Industrial",
        "criticality": "High",
        "base_rpm": 1190,
    },
]


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """
    Create a SQLite connection with foreign keys enabled.
    """

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def generate_time_vector() -> np.ndarray:
    """
    Generate the time vector used by all synthetic measurements.
    """

    return np.arange(N_SAMPLES) / SAMPLING_RATE_HZ


def generate_vibration_signal(
    scenario_name: str,
    rpm: float,
    degradation_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic vibration velocity signal in mm/s.

    The signal is intentionally simple, but follows vibration-analysis logic:

    normal_operation:
        dominant 1x component, low random noise.

    carpet_lubrication_issue:
        base 1x component plus broadband high-frequency noise and micro-impacts.

    structural_looseness:
        stronger low-frequency content with 0.5x, 1x, 1.5x, 2x and 3x components.
    """

    time = generate_time_vector()
    rotational_frequency_hz = rpm / 60.0

    phase = rng.uniform(0, 2 * math.pi)

    base_signal = 1.2 * np.sin(
        2 * math.pi * rotational_frequency_hz * time + phase
    )

    noise = rng.normal(0, 0.08, size=N_SAMPLES)

    if scenario_name == "normal_operation":
        signal = base_signal + noise

    elif scenario_name == "carpet_lubrication_issue":
        high_frequency_noise = rng.normal(
            0,
            0.18 + 0.55 * degradation_level,
            size=N_SAMPLES,
        )

        carrier_1 = np.sin(2 * math.pi * 180 * time + rng.uniform(0, 2 * math.pi))
        carrier_2 = np.sin(2 * math.pi * 260 * time + rng.uniform(0, 2 * math.pi))
        carrier_3 = np.sin(2 * math.pi * 340 * time + rng.uniform(0, 2 * math.pi))

        broadband_band = high_frequency_noise * (
            0.45 * carrier_1 + 0.35 * carrier_2 + 0.20 * carrier_3
        )

        micro_impacts = np.zeros(N_SAMPLES)
        n_impacts = int(6 + 20 * degradation_level)

        impact_positions = rng.choice(
            np.arange(20, N_SAMPLES - 20),
            size=n_impacts,
            replace=False,
        )

        for position in impact_positions:
            impact_width = rng.integers(3, 10)
            impact_amplitude = rng.uniform(0.4, 1.2) * degradation_level
            decay = np.exp(-np.linspace(0, 3, impact_width))
            micro_impacts[position : position + impact_width] += (
                impact_amplitude * decay
            )

        signal = (
            base_signal
            + noise
            + broadband_band
            + micro_impacts
        )

    elif scenario_name == "structural_looseness":
        half_x = 0.55 * degradation_level * np.sin(
            2 * math.pi * 0.5 * rotational_frequency_hz * time
            + rng.uniform(0, 2 * math.pi)
        )
        one_and_half_x = 0.45 * degradation_level * np.sin(
            2 * math.pi * 1.5 * rotational_frequency_hz * time
            + rng.uniform(0, 2 * math.pi)
        )
        two_x = 0.75 * degradation_level * np.sin(
            2 * math.pi * 2.0 * rotational_frequency_hz * time
            + rng.uniform(0, 2 * math.pi)
        )
        three_x = 0.45 * degradation_level * np.sin(
            2 * math.pi * 3.0 * rotational_frequency_hz * time
            + rng.uniform(0, 2 * math.pi)
        )

        looseness_noise = rng.normal(
            0,
            0.10 + 0.12 * degradation_level,
            size=N_SAMPLES,
        )

        signal = (
            base_signal
            + half_x
            + one_and_half_x
            + two_x
            + three_x
            + looseness_noise
        )

    else:
        raise ValueError(f"Unknown scenario_name: {scenario_name}")

    return signal.astype(float)


def calculate_fft(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate single-sided FFT amplitude spectrum.
    """

    signal_without_mean = signal - np.mean(signal)

    window = np.hanning(len(signal_without_mean))
    windowed_signal = signal_without_mean * window

    fft_values = np.fft.rfft(windowed_signal)
    frequencies = np.fft.rfftfreq(len(windowed_signal), d=1 / SAMPLING_RATE_HZ)

    amplitude = np.abs(fft_values) / len(windowed_signal)
    amplitude = 2 * amplitude

    return frequencies, amplitude


def get_band_energy(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    """
    Calculate energy inside a frequency band.
    """

    mask = (frequencies >= lower_hz) & (frequencies < upper_hz)

    if not np.any(mask):
        return 0.0

    return float(np.sum(amplitude[mask] ** 2))


def get_amplitude_near_frequency(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency_hz: float,
    tolerance_hz: float = 1.0,
) -> float:
    """
    Get the maximum amplitude around a target frequency.
    """

    mask = (
        frequencies >= target_frequency_hz - tolerance_hz
    ) & (
        frequencies <= target_frequency_hz + tolerance_hz
    )

    if not np.any(mask):
        return 0.0

    return float(np.max(amplitude[mask]))


def calculate_spectral_features(signal: np.ndarray, rpm: float) -> dict[str, float]:
    """
    Calculate physics-informed spectral features from the raw signal.
    """

    frequencies, amplitude = calculate_fft(signal)

    rotational_frequency_hz = rpm / 60.0

    low_frequency_energy = get_band_energy(frequencies, amplitude, 0.5, 80.0)
    mid_frequency_energy = get_band_energy(frequencies, amplitude, 80.0, 180.0)
    high_frequency_energy = get_band_energy(frequencies, amplitude, 180.0, 450.0)
    broadband_energy = get_band_energy(frequencies, amplitude, 80.0, 450.0)

    amplitude_0_5x = get_amplitude_near_frequency(
        frequencies,
        amplitude,
        0.5 * rotational_frequency_hz,
    )
    amplitude_1x = get_amplitude_near_frequency(
        frequencies,
        amplitude,
        1.0 * rotational_frequency_hz,
    )
    amplitude_1_5x = get_amplitude_near_frequency(
        frequencies,
        amplitude,
        1.5 * rotational_frequency_hz,
    )
    amplitude_2x = get_amplitude_near_frequency(
        frequencies,
        amplitude,
        2.0 * rotational_frequency_hz,
    )
    amplitude_3x = get_amplitude_near_frequency(
        frequencies,
        amplitude,
        3.0 * rotational_frequency_hz,
    )

    harmonic_ratio = (amplitude_2x + amplitude_3x) / max(amplitude_1x, 1e-9)
    subharmonic_ratio = (amplitude_0_5x + amplitude_1_5x) / max(amplitude_1x, 1e-9)

    high_frequency_ratio = high_frequency_energy / max(
        low_frequency_energy + mid_frequency_energy + high_frequency_energy,
        1e-9,
    )

    high_frequency_mask = (frequencies >= 180.0) & (frequencies <= 450.0)

    if np.any(high_frequency_mask):
        high_frequency_amplitude = amplitude[high_frequency_mask]
        spectral_floor_level = float(np.median(high_frequency_amplitude))
        spectral_flatness = float(
            np.exp(np.mean(np.log(high_frequency_amplitude + 1e-12)))
            / max(np.mean(high_frequency_amplitude), 1e-12)
        )
    else:
        spectral_floor_level = 0.0
        spectral_flatness = 0.0

    positive_mask = frequencies > 0

    if np.any(positive_mask):
        dominant_index = int(np.argmax(amplitude[positive_mask]))
        positive_frequencies = frequencies[positive_mask]
        positive_amplitude = amplitude[positive_mask]

        dominant_frequency_hz = float(positive_frequencies[dominant_index])
        dominant_amplitude = float(positive_amplitude[dominant_index])

        spectral_centroid_hz = float(
            np.sum(positive_frequencies * positive_amplitude)
            / max(np.sum(positive_amplitude), 1e-9)
        )

        centered = positive_amplitude - np.mean(positive_amplitude)
        spectral_kurtosis = float(
            np.mean(centered**4)
            / max(np.var(positive_amplitude) ** 2, 1e-9)
        )
    else:
        dominant_frequency_hz = 0.0
        dominant_amplitude = 0.0
        spectral_centroid_hz = 0.0
        spectral_kurtosis = 0.0

    return {
        "rotational_frequency_hz": float(rotational_frequency_hz),
        "low_frequency_energy": float(low_frequency_energy),
        "mid_frequency_energy": float(mid_frequency_energy),
        "high_frequency_energy": float(high_frequency_energy),
        "broadband_energy": float(broadband_energy),
        "spectral_floor_level": float(spectral_floor_level),
        "spectral_flatness": float(spectral_flatness),
        "high_frequency_ratio": float(high_frequency_ratio),
        "amplitude_0_5x": float(amplitude_0_5x),
        "amplitude_1x": float(amplitude_1x),
        "amplitude_1_5x": float(amplitude_1_5x),
        "amplitude_2x": float(amplitude_2x),
        "amplitude_3x": float(amplitude_3x),
        "harmonic_ratio": float(harmonic_ratio),
        "subharmonic_ratio": float(subharmonic_ratio),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "dominant_amplitude": float(dominant_amplitude),
        "spectral_centroid_hz": float(spectral_centroid_hz),
        "spectral_kurtosis": float(spectral_kurtosis),
    }


def normalize_score(value: float, low_reference: float, high_reference: float) -> float:
    """
    Normalize a feature into a 0-1 score using simple reference limits.
    """

    if high_reference <= low_reference:
        return 0.0

    score = (value - low_reference) / (high_reference - low_reference)

    return float(np.clip(score, 0.0, 1.0))


def calculate_condition_scores(features: dict[str, float]) -> dict[str, Any]:
    """
    Calculate carpet and looseness scores using physics-informed logic.
    """

    carpet_score = np.mean(
        [
            normalize_score(features["high_frequency_energy"], 0.015, 0.120),
            normalize_score(features["broadband_energy"], 0.020, 0.180),
            normalize_score(features["spectral_floor_level"], 0.001, 0.012),
            normalize_score(features["high_frequency_ratio"], 0.10, 0.55),
            normalize_score(features["spectral_flatness"], 0.15, 0.75),
        ]
    )

    looseness_score = np.mean(
        [
            normalize_score(features["low_frequency_energy"], 0.50, 2.20),
            normalize_score(features["harmonic_ratio"], 0.10, 0.85),
            normalize_score(features["subharmonic_ratio"], 0.05, 0.70),
            normalize_score(features["amplitude_0_5x"], 0.02, 0.40),
            normalize_score(features["amplitude_1_5x"], 0.02, 0.35),
            normalize_score(features["amplitude_2x"], 0.03, 0.55),
            normalize_score(features["amplitude_3x"], 0.02, 0.40),
        ]
    )

    carpet_score = float(np.clip(carpet_score, 0.0, 1.0))
    looseness_score = float(np.clip(looseness_score, 0.0, 1.0))

    overall_anomaly_score = float(max(carpet_score, looseness_score))

    anomaly_probability = float(
        1 / (1 + math.exp(-8 * (overall_anomaly_score - 0.50)))
    )

    if carpet_score >= 0.55 and looseness_score >= 0.55:
        predicted_condition = "mixed_fault_pattern"
    elif carpet_score >= looseness_score and carpet_score >= 0.45:
        predicted_condition = "carpet_lubrication_issue"
    elif looseness_score > carpet_score and looseness_score >= 0.45:
        predicted_condition = "structural_looseness"
    else:
        predicted_condition = "normal_operation"

    if overall_anomaly_score >= 0.72:
        severity = "Critical"
    elif overall_anomaly_score >= 0.45:
        severity = "Attention"
    else:
        severity = "Normal"

    diagnostic_explanation = build_diagnostic_explanation(
        predicted_condition=predicted_condition,
        carpet_score=carpet_score,
        looseness_score=looseness_score,
        features=features,
    )

    recommended_action = build_recommended_action(
        predicted_condition=predicted_condition,
        severity=severity,
    )

    return {
        "carpet_score": carpet_score,
        "looseness_score": looseness_score,
        "overall_anomaly_score": overall_anomaly_score,
        "anomaly_probability": anomaly_probability,
        "predicted_condition": predicted_condition,
        "severity": severity,
        "diagnostic_explanation": diagnostic_explanation,
        "recommended_action": recommended_action,
    }


def build_diagnostic_explanation(
    predicted_condition: str,
    carpet_score: float,
    looseness_score: float,
    features: dict[str, float],
) -> str:
    """
    Build a deterministic explanation for the diagnostic result.
    """

    if predicted_condition == "carpet_lubrication_issue":
        return (
            "The signal shows elevated broadband and high-frequency energy, "
            "with increased spectral floor. This pattern is compatible with "
            "a lubrication-related carpet effect."
        )

    if predicted_condition == "structural_looseness":
        return (
            "The signal shows increased low-frequency energy with relevant "
            "sub-harmonic and harmonic components around 0.5x, 1.5x, 2x and 3x. "
            "This pattern is compatible with structural looseness."
        )

    if predicted_condition == "mixed_fault_pattern":
        return (
            "The signal combines broadband high-frequency elevation and "
            "low-frequency harmonic/sub-harmonic components. This suggests "
            "a mixed fault pattern requiring inspection."
        )

    return (
        "The signal is dominated by the rotational component and does not show "
        "relevant broadband, harmonic or sub-harmonic abnormality."
    )


def build_recommended_action(predicted_condition: str, severity: str) -> str:
    """
    Build a rule-based recommended action for the calculated condition.
    """

    if predicted_condition == "carpet_lubrication_issue":
        if severity == "Critical":
            return (
                "Prioritize lubrication inspection, check lubricant condition, "
                "verify contamination, and inspect bearing condition."
            )

        return (
            "Schedule lubrication check, review lubrication interval, and monitor "
            "high-frequency broadband trend."
        )

    if predicted_condition == "structural_looseness":
        if severity == "Critical":
            return (
                "Prioritize inspection of foundation, bolts, supports, coupling "
                "alignment and mechanical clearances."
            )

        return (
            "Schedule mechanical inspection and monitor harmonic and sub-harmonic "
            "components over the next readings."
        )

    if predicted_condition == "mixed_fault_pattern":
        return (
            "Perform combined lubrication and mechanical inspection. Validate "
            "bearing condition, foundation, bolts, alignment and operating load."
        )

    return (
        "Continue normal monitoring routine and keep vibration trend under observation."
    )


def calculate_time_domain_metrics(signal: np.ndarray) -> dict[str, float]:
    """
    Calculate time-domain vibration metrics.
    """

    rms_velocity_mm_s = float(np.sqrt(np.mean(signal**2)))
    peak_velocity_mm_s = float(np.max(np.abs(signal)))
    crest_factor = float(peak_velocity_mm_s / max(rms_velocity_mm_s, 1e-9))

    return {
        "rms_velocity_mm_s": rms_velocity_mm_s,
        "peak_velocity_mm_s": peak_velocity_mm_s,
        "crest_factor": crest_factor,
    }


def insert_scenarios(connection: sqlite3.Connection) -> dict[str, int]:
    """
    Insert diagnostic scenarios and return their IDs.
    """

    cursor = connection.cursor()

    for scenario in SCENARIOS:
        cursor.execute(
            """
            INSERT INTO scenarios (
                scenario_name,
                scenario_label,
                scenario_description,
                expected_fault_pattern
            )
            VALUES (?, ?, ?, ?);
            """,
            (
                scenario["scenario_name"],
                scenario["scenario_label"],
                scenario["scenario_description"],
                scenario["expected_fault_pattern"],
            ),
        )

    connection.commit()

    rows = cursor.execute(
        """
        SELECT scenario_id, scenario_name
        FROM scenarios;
        """
    ).fetchall()

    return {scenario_name: scenario_id for scenario_id, scenario_name in rows}


def insert_assets(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """
    Insert assets and return their database IDs plus metadata.
    """

    cursor = connection.cursor()

    for asset in ASSETS:
        cursor.execute(
            """
            INSERT INTO assets (
                asset_code,
                asset_name,
                asset_type,
                plant_area,
                manufacturer,
                criticality
            )
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                asset["asset_code"],
                asset["asset_name"],
                asset["asset_type"],
                asset["plant_area"],
                asset["manufacturer"],
                asset["criticality"],
            ),
        )

    connection.commit()

    rows = cursor.execute(
        """
        SELECT asset_id, asset_code
        FROM assets;
        """
    ).fetchall()

    asset_id_by_code = {asset_code: asset_id for asset_id, asset_code in rows}

    return {
        asset["asset_code"]: {
            "asset_id": asset_id_by_code[asset["asset_code"]],
            "base_rpm": asset["base_rpm"],
            "asset_name": asset["asset_name"],
        }
        for asset in ASSETS
    }


def choose_operating_scenario(day_index: int, asset_index: int) -> tuple[str, float]:
    """
    Choose scenario and degradation level across time.

    This creates a realistic demo history:
    - early days mostly normal
    - middle days start showing attention patterns
    - later days show stronger degradation for selected assets
    """

    progression = day_index / max(N_DAYS - 1, 1)

    if asset_index == 0:
        if progression < 0.35:
            return "normal_operation", 0.10
        return "carpet_lubrication_issue", min(1.0, 0.25 + progression)

    if asset_index == 1:
        if progression < 0.45:
            return "normal_operation", 0.12
        return "structural_looseness", min(1.0, 0.20 + progression)

    if progression < 0.65:
        return "normal_operation", 0.08

    if day_index % 2 == 0:
        return "carpet_lubrication_issue", min(0.85, 0.15 + progression)

    return "structural_looseness", min(0.85, 0.15 + progression)


def insert_measurement(
    connection: sqlite3.Connection,
    asset_id: int,
    scenario_id: int,
    timestamp: datetime,
    rpm: float,
    metrics: dict[str, float],
    operating_condition: str,
) -> int:
    """
    Insert one vibration measurement and return its ID.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO vibration_measurements (
            asset_id,
            scenario_id,
            timestamp,
            collection_date,
            sensor_position,
            axis,
            sampling_rate_hz,
            duration_seconds,
            rpm,
            rms_velocity_mm_s,
            peak_velocity_mm_s,
            crest_factor,
            temperature_c,
            operating_condition,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            asset_id,
            scenario_id,
            timestamp.isoformat(timespec="seconds"),
            timestamp.date().isoformat(),
            "Drive end bearing",
            "radial",
            SAMPLING_RATE_HZ,
            DURATION_SECONDS,
            rpm,
            metrics["rms_velocity_mm_s"],
            metrics["peak_velocity_mm_s"],
            metrics["crest_factor"],
            42.0 + metrics["rms_velocity_mm_s"] * 1.5,
            operating_condition,
            "Synthetic physics-informed vibration measurement.",
        ),
    )

    connection.commit()

    return int(cursor.lastrowid)


def insert_raw_samples(
    connection: sqlite3.Connection,
    measurement_id: int,
    signal: np.ndarray,
) -> None:
    """
    Insert raw vibration samples for a measurement.
    """

    time = generate_time_vector()

    rows = [
        (
            measurement_id,
            int(sample_index),
            float(time_seconds),
            None,
            float(velocity_mm_s),
        )
        for sample_index, (time_seconds, velocity_mm_s)
        in enumerate(zip(time, signal))
    ]

    connection.executemany(
        """
        INSERT INTO vibration_raw_samples (
            measurement_id,
            sample_index,
            time_seconds,
            acceleration_g,
            velocity_mm_s
        )
        VALUES (?, ?, ?, ?, ?);
        """,
        rows,
    )

    connection.commit()


def insert_spectral_features(
    connection: sqlite3.Connection,
    measurement_id: int,
    features: dict[str, float],
) -> None:
    """
    Insert spectral features calculated from raw vibration signal.
    """

    connection.execute(
        """
        INSERT INTO spectral_features (
            measurement_id,
            rotational_frequency_hz,
            low_frequency_energy,
            mid_frequency_energy,
            high_frequency_energy,
            broadband_energy,
            spectral_floor_level,
            spectral_flatness,
            high_frequency_ratio,
            amplitude_0_5x,
            amplitude_1x,
            amplitude_1_5x,
            amplitude_2x,
            amplitude_3x,
            harmonic_ratio,
            subharmonic_ratio,
            dominant_frequency_hz,
            dominant_amplitude,
            spectral_centroid_hz,
            spectral_kurtosis
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            measurement_id,
            features["rotational_frequency_hz"],
            features["low_frequency_energy"],
            features["mid_frequency_energy"],
            features["high_frequency_energy"],
            features["broadband_energy"],
            features["spectral_floor_level"],
            features["spectral_flatness"],
            features["high_frequency_ratio"],
            features["amplitude_0_5x"],
            features["amplitude_1x"],
            features["amplitude_1_5x"],
            features["amplitude_2x"],
            features["amplitude_3x"],
            features["harmonic_ratio"],
            features["subharmonic_ratio"],
            features["dominant_frequency_hz"],
            features["dominant_amplitude"],
            features["spectral_centroid_hz"],
            features["spectral_kurtosis"],
        ),
    )

    connection.commit()


def insert_ml_diagnostics(
    connection: sqlite3.Connection,
    measurement_id: int,
    diagnostics: dict[str, Any],
) -> None:
    """
    Insert diagnostics calculated by the physics-informed scoring layer.
    """

    connection.execute(
        """
        INSERT INTO ml_diagnostics (
            measurement_id,
            carpet_score,
            looseness_score,
            overall_anomaly_score,
            anomaly_probability,
            predicted_condition,
            severity,
            diagnostic_method,
            model_name,
            model_version,
            diagnostic_explanation,
            recommended_action
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            measurement_id,
            diagnostics["carpet_score"],
            diagnostics["looseness_score"],
            diagnostics["overall_anomaly_score"],
            diagnostics["anomaly_probability"],
            diagnostics["predicted_condition"],
            diagnostics["severity"],
            "physics_informed_rule_based",
            "physics_informed_scoring",
            "v1",
            diagnostics["diagnostic_explanation"],
            diagnostics["recommended_action"],
        ),
    )

    connection.commit()


def insert_feature_diagnostics(
    connection: sqlite3.Connection,
    measurement_id: int,
    features: dict[str, float],
    diagnostics: dict[str, Any],
) -> None:
    """
    Insert feature-level diagnostic explanations.
    """

    rows = []

    if diagnostics["predicted_condition"] in {
        "carpet_lubrication_issue",
        "mixed_fault_pattern",
    }:
        rows.extend(
            [
                (
                    measurement_id,
                    "high_frequency_energy",
                    features["high_frequency_energy"],
                    0.120,
                    diagnostics["severity"],
                    "High-frequency energy contributes to carpet/lubrication issue detection.",
                ),
                (
                    measurement_id,
                    "spectral_floor_level",
                    features["spectral_floor_level"],
                    0.012,
                    diagnostics["severity"],
                    "Elevated spectral floor indicates broadband vibration increase.",
                ),
            ]
        )

    if diagnostics["predicted_condition"] in {
        "structural_looseness",
        "mixed_fault_pattern",
    }:
        rows.extend(
            [
                (
                    measurement_id,
                    "harmonic_ratio",
                    features["harmonic_ratio"],
                    0.850,
                    diagnostics["severity"],
                    "Harmonic content contributes to structural looseness detection.",
                ),
                (
                    measurement_id,
                    "subharmonic_ratio",
                    features["subharmonic_ratio"],
                    0.700,
                    diagnostics["severity"],
                    "Sub-harmonic content contributes to structural looseness detection.",
                ),
            ]
        )

    if not rows:
        rows.append(
            (
                measurement_id,
                "overall_anomaly_score",
                diagnostics["overall_anomaly_score"],
                0.450,
                "Normal",
                "No relevant abnormal vibration pattern detected.",
            )
        )

    connection.executemany(
        """
        INSERT INTO feature_diagnostics (
            measurement_id,
            feature_name,
            feature_value,
            reference_value,
            diagnostic_flag,
            diagnostic_reason
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        rows,
    )

    connection.commit()


def seed_database(db_path: Path = DATABASE_PATH) -> None:
    """
    Recreate and populate the demo database.
    """

    reset_database(db_path)

    rng = np.random.default_rng(RANDOM_SEED)

    with get_connection(db_path) as connection:
        scenario_ids = insert_scenarios(connection)
        asset_metadata = insert_assets(connection)

        asset_items = list(asset_metadata.items())

        for day_index in range(N_DAYS):
            for measurement_slot in range(MEASUREMENTS_PER_DAY):
                for asset_index, (_, asset_info) in enumerate(asset_items):
                    scenario_name, degradation_level = choose_operating_scenario(
                        day_index=day_index,
                        asset_index=asset_index,
                    )

                    timestamp = (
                        START_DATE
                        + timedelta(days=day_index)
                        + timedelta(hours=measurement_slot * 4)
                    )

                    rpm_variation = rng.normal(0, 8)
                    rpm = float(asset_info["base_rpm"] + rpm_variation)

                    signal = generate_vibration_signal(
                        scenario_name=scenario_name,
                        rpm=rpm,
                        degradation_level=degradation_level,
                        rng=rng,
                    )

                    metrics = calculate_time_domain_metrics(signal)
                    features = calculate_spectral_features(signal, rpm)
                    diagnostics = calculate_condition_scores(features)

                    measurement_id = insert_measurement(
                        connection=connection,
                        asset_id=asset_info["asset_id"],
                        scenario_id=scenario_ids[scenario_name],
                        timestamp=timestamp,
                        rpm=rpm,
                        metrics=metrics,
                        operating_condition=scenario_name,
                    )

                    insert_raw_samples(
                        connection=connection,
                        measurement_id=measurement_id,
                        signal=signal,
                    )

                    insert_spectral_features(
                        connection=connection,
                        measurement_id=measurement_id,
                        features=features,
                    )

                    insert_ml_diagnostics(
                        connection=connection,
                        measurement_id=measurement_id,
                        diagnostics=diagnostics,
                    )

                    insert_feature_diagnostics(
                        connection=connection,
                        measurement_id=measurement_id,
                        features=features,
                        diagnostics=diagnostics,
                    )


if __name__ == "__main__":
    seed_database()

    with get_connection() as conn:
        measurements_count = conn.execute(
            "SELECT COUNT(*) FROM vibration_measurements;"
        ).fetchone()[0]

        raw_samples_count = conn.execute(
            "SELECT COUNT(*) FROM vibration_raw_samples;"
        ).fetchone()[0]

        diagnostics_count = conn.execute(
            "SELECT COUNT(*) FROM ml_diagnostics;"
        ).fetchone()[0]

    print("Database seeded successfully.")
    print(f"Measurements: {measurements_count}")
    print(f"Raw samples: {raw_samples_count}")
    print(f"Diagnostics: {diagnostics_count}")