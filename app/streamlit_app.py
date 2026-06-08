"""
Streamlit app for the Hybrid AI Predictive Maintenance project.

This version focuses the user experience on two flows:
1. Monitoring: latest condition by equipment using a selected collection date,
   raw vibration samples, FFT, PSD, and deterministic action plans.
2. Semantic AI Query: natural-language database consultation using controlled routing,
   approved SQL templates, embedding fallback, cosine similarity, and SQL Guard.

This version does not depend on paid external generative AI services. The current action plan
layer is deterministic and rule-based.
"""

from __future__ import annotations

from pathlib import Path
from html import escape
import sys
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import (  # noqa: E402
    get_available_collection_dates,
    get_full_diagnostic_view,
    get_latest_measurements_until_date,
    get_raw_vibration_samples,
    get_scenarios,
)
from src.llm.embedding_router import suggest_approved_questions  # noqa: E402
from src.llm.semantic_query_router import (  # noqa: E402
    get_supported_demo_questions,
    get_supported_query_examples,
    route_prompt_to_sql,
)
from src.llm.text_to_sql import run_text_to_sql  # noqa: E402


APP_DIR = Path(__file__).resolve().parent
PAGE_ICON_PATH = APP_DIR / "page_icon.png"
SIDEBAR_BACKGROUND_PATH = APP_DIR / "background_sidebar.png"


st.set_page_config(
    page_title="Semantic AI Predictive Maintenance",
    page_icon=str(PAGE_ICON_PATH),
    layout="wide",
)

def get_base64_image(image_path: Path) -> str:
    """Encode a local image file as base64 for CSS usage."""
    if not image_path.exists():
        return ""

    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def apply_sidebar_background() -> None:
    """Apply a custom background image to the Streamlit sidebar."""
    encoded_image = get_base64_image(SIDEBAR_BACKGROUND_PATH)

    if not encoded_image:
        return

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image:
                linear-gradient(
                    rgba(5, 18, 32, 0.78),
                    rgba(5, 18, 32, 0.90)
                ),
                url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: #F8FAFC;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: rgba(248, 250, 252, 0.25);
        }}

        /* Selectbox closed field text */
        [data-testid="stSidebar"] [data-baseweb="select"] div {{
            color: #111827 !important;
        }}

        /* Selectbox selected value */
        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: #111827 !important;
        }}

        /* Selectbox input */
        [data-testid="stSidebar"] [data-baseweb="select"] input {{
            color: #111827 !important;
        }}

        /* Selectbox white container */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            color: #111827 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float when possible."""
    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_existing_value(row: pd.Series, fields: list[str], default: str = "") -> str:
    """Return the first non-empty value from a row using candidate field names."""
    for field in fields:
        if field in row.index and pd.notna(row.get(field)):
            value = str(row.get(field))
            if value.strip():
                return value
    return default


def normalize_card_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy-friendly aliases expected by the card renderer.

    The database evolved from aggregated diagnostic columns to a raw-vibration
    scoring schema. This function keeps the UI robust while the SQL layer and
    old tests still use some legacy names.
    """
    if df.empty:
        return df.copy()

    normalized_df = df.copy()

    alias_pairs = {
        "rms_velocity": "rms_velocity_mm_s",
        "peak_velocity": "peak_velocity_mm_s",
        "anomaly_score": "overall_anomaly_score",
        "predicted_label": "predicted_condition",
        "explanation": "diagnostic_explanation",
        "location": "plant_area",
    }

    for legacy_column, new_column in alias_pairs.items():
        if legacy_column not in normalized_df.columns and new_column in normalized_df.columns:
            normalized_df[legacy_column] = normalized_df[new_column]

    if "severity_level" not in normalized_df.columns and "severity" in normalized_df.columns:
        normalized_df["severity_level"] = normalized_df["severity"].map(
            {
                "Critical": "High",
                "Attention": "Medium",
                "Normal": "Low",
            }
        ).fillna(normalized_df["severity"])

    if "asset_code" not in normalized_df.columns and "asset_name" in normalized_df.columns:
        normalized_df["asset_code"] = normalized_df["asset_name"]

    return normalized_df


# -----------------------------------------------------------------------------
# Raw vibration signal processing and charts
# -----------------------------------------------------------------------------


def calculate_single_sided_fft(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a single-sided FFT amplitude spectrum."""
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([])

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    n_samples = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate_hz)

    amplitudes = np.abs(fft_values) / n_samples

    if len(amplitudes) > 2:
        amplitudes[1:-1] = 2.0 * amplitudes[1:-1]

    return frequencies, amplitudes


def calculate_welch_like_psd(
    signal: np.ndarray,
    sampling_rate_hz: float,
    segment_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a compact Welch-like PSD without requiring scipy."""
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([])

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    n_samples = len(signal)

    if n_samples < segment_size:
        segment_size = n_samples

    if segment_size < 8:
        return np.array([]), np.array([])

    step = max(segment_size // 2, 1)
    window = np.hanning(segment_size)
    window_power = np.sum(window**2)

    psd_segments = []

    for start in range(0, n_samples - segment_size + 1, step):
        segment = signal[start : start + segment_size]
        segment = segment - np.mean(segment)
        windowed_segment = segment * window

        fft_values = np.fft.rfft(windowed_segment)
        psd = (np.abs(fft_values) ** 2) / (sampling_rate_hz * window_power)
        psd_segments.append(psd)

    if not psd_segments:
        return np.array([]), np.array([])

    psd_mean = np.mean(psd_segments, axis=0)
    frequencies = np.fft.rfftfreq(segment_size, d=1.0 / sampling_rate_hz)

    return frequencies, psd_mean


def build_time_series_chart(
    raw_samples: pd.DataFrame,
    asset_code: str,
) -> go.Figure:
    """Build time-domain vibration chart from raw samples."""
    fig = go.Figure()

    if raw_samples.empty:
        fig.update_layout(
            title="Raw vibration signal unavailable",
            height=220,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=raw_samples["time_seconds"],
            y=raw_samples["velocity_mm_s"],
            mode="lines",
            name="Velocity",
        )
    )

    fig.update_layout(
        title=f"{asset_code} | Raw vibration signal",
        xaxis_title="Time [s]",
        yaxis_title="Velocity [mm/s]",
        height=240,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )

    return fig


def build_fft_chart(
    raw_samples: pd.DataFrame,
    sampling_rate_hz: float,
    rpm: float,
    asset_code: str,
    predicted_condition: str,
) -> go.Figure:
    """Build FFT chart with 0.5x, 1x, 1.5x, 2x and 3x rotational markers."""
    fig = go.Figure()

    if raw_samples.empty:
        fig.update_layout(
            title="FFT unavailable",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        return fig

    signal = raw_samples["velocity_mm_s"].to_numpy()
    frequencies, amplitudes = calculate_single_sided_fft(
        signal=signal,
        sampling_rate_hz=sampling_rate_hz,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=amplitudes,
            mode="lines",
            name="FFT amplitude",
        )
    )

    rotational_frequency_hz = rpm / 60.0 if rpm else None

    if rotational_frequency_hz and len(frequencies) > 0:
        markers = [
            (0.5, "0.5x"),
            (1.0, "1x"),
            (1.5, "1.5x"),
            (2.0, "2x"),
            (3.0, "3x"),
        ]

        for multiplier, label in markers:
            marker_frequency = rotational_frequency_hz * multiplier

            if marker_frequency <= frequencies.max():
                fig.add_vline(
                    x=marker_frequency,
                    line_dash="dash",
                    annotation_text=label,
                    annotation_position="top",
                )

    max_frequency_to_show = min(300, float(frequencies.max())) if len(frequencies) else 300

    fig.update_layout(
        title=f"{asset_code} | FFT spectrum | {predicted_condition}",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude [mm/s]",
        xaxis_range=[0, max_frequency_to_show],
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )

    return fig


def build_psd_chart(
    raw_samples: pd.DataFrame,
    sampling_rate_hz: float,
    asset_code: str,
) -> go.Figure:
    """Build PSD chart to show broadband spectral floor behavior."""
    fig = go.Figure()

    if raw_samples.empty:
        fig.update_layout(
            title="PSD unavailable",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        return fig

    signal = raw_samples["velocity_mm_s"].to_numpy()
    frequencies, psd = calculate_welch_like_psd(
        signal=signal,
        sampling_rate_hz=sampling_rate_hz,
        segment_size=256,
    )

    if len(frequencies) == 0:
        fig.update_layout(
            title="PSD unavailable",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=psd,
            mode="lines",
            name="PSD",
        )
    )

    max_frequency_to_show = min(500, float(frequencies.max()))

    fig.update_layout(
        title=f"{asset_code} | PSD / broadband energy",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Power spectral density",
        xaxis_range=[0, max_frequency_to_show],
        yaxis_type="log",
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )

    return fig


def render_vibration_charts_for_measurement(
    measurement_row: pd.Series,
    chart_key_prefix: str,
) -> None:
    """Render raw signal, FFT and PSD charts for a measurement."""
    if "measurement_id" not in measurement_row.index or pd.isna(measurement_row.get("measurement_id")):
        st.info("Raw vibration charts require a measurement_id.")
        return

    measurement_id = int(measurement_row["measurement_id"])
    asset_code = first_existing_value(
        measurement_row,
        ["asset_code", "asset_name"],
        default="Asset",
    )
    sampling_rate_hz = safe_float(measurement_row.get("sampling_rate_hz"), 1024.0)
    rpm = safe_float(measurement_row.get("rpm"), 1800.0)
    predicted_condition = first_existing_value(
        measurement_row,
        ["predicted_condition", "predicted_label", "scenario_name"],
        default="unknown_condition",
    )

    raw_samples = get_raw_vibration_samples(measurement_id)

    chart_tab_1, chart_tab_2, chart_tab_3 = st.tabs(
        [
            "Raw signal",
            "FFT",
            "PSD",
        ]
    )

    with chart_tab_1:
        st.plotly_chart(
            build_time_series_chart(
                raw_samples=raw_samples,
                asset_code=asset_code,
            ),
            use_container_width=True,
            key=f"{chart_key_prefix}_raw",
        )

    with chart_tab_2:
        st.plotly_chart(
            build_fft_chart(
                raw_samples=raw_samples,
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                asset_code=asset_code,
                predicted_condition=predicted_condition,
            ),
            use_container_width=True,
            key=f"{chart_key_prefix}_fft",
        )

    with chart_tab_3:
        st.plotly_chart(
            build_psd_chart(
                raw_samples=raw_samples,
                sampling_rate_hz=sampling_rate_hz,
                asset_code=asset_code,
            ),
            use_container_width=True,
            key=f"{chart_key_prefix}_psd",
        )


# -----------------------------------------------------------------------------
# Data loading and global filters
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_scenarios() -> pd.DataFrame:
    """Load scenarios from SQLite."""
    return get_scenarios()


@st.cache_data(show_spinner=False)
def load_diagnostic_data() -> pd.DataFrame:
    """Load full diagnostic data from SQLite for Chat and trend views."""
    return normalize_card_columns(get_full_diagnostic_view())


@st.cache_data(show_spinner=False)
def load_available_dates() -> pd.DataFrame:
    """Load available collection dates from SQLite.

    The query helper may return either a DataFrame or a plain list, depending
    on the current implementation. The Streamlit layer normalizes both cases
    to a DataFrame with one column: collection_date.
    """

    available_dates = get_available_collection_dates()

    if isinstance(available_dates, pd.DataFrame):
        if "collection_date" in available_dates.columns:
            normalized_dates = available_dates[["collection_date"]].copy()
        elif len(available_dates.columns) > 0:
            normalized_dates = available_dates.iloc[:, [0]].copy()
            normalized_dates.columns = ["collection_date"]
        else:
            normalized_dates = pd.DataFrame(columns=["collection_date"])
    else:
        normalized_dates = pd.DataFrame(
            {"collection_date": [str(date) for date in available_dates]}
        )

    normalized_dates = normalized_dates.dropna()
    normalized_dates["collection_date"] = normalized_dates["collection_date"].astype(str)
    normalized_dates = normalized_dates.drop_duplicates().sort_values("collection_date")

    return normalized_dates.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_latest_measurements_until_date(collection_date: str) -> pd.DataFrame:
    """Load the latest measurement for each equipment until the selected date."""
    return normalize_card_columns(get_latest_measurements_until_date(collection_date))


def render_header() -> None:
    """Render the app header."""
    st.title("Semantic AI for Predictive Maintenance")
    st.markdown(
        """
        A practical lab for combining physics-informed condition monitoring,
        Semantic AI database interaction, SQL safety, raw vibration signal
        analysis, and deterministic action planning in an industrial predictive
        maintenance scenario.
        """
    )


def render_sidebar(scenarios_df: pd.DataFrame) -> tuple[str, str | None]:
    """Render sidebar filters and return selected scenario and collection date."""
    st.sidebar.header("Monitoring filter")

    scenario_options = ["All scenarios"] + scenarios_df["scenario_label"].tolist()

    selected_scenario_label = st.sidebar.selectbox(
        "Choose an industrial scenario",
        scenario_options,
    )

    available_dates = load_available_dates()
    selected_collection_date = None

    if not available_dates.empty and "collection_date" in available_dates.columns:
        date_options = available_dates["collection_date"].astype(str).tolist()
        selected_collection_date = st.sidebar.selectbox(
            "Collection date",
            options=date_options,
            index=len(date_options) - 1,
            help="Monitoring cards use the latest available reading for each equipment up to this date.",
        )
    else:
        st.sidebar.warning("No collection dates found in the database.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Initial scenarios")
    st.sidebar.markdown(
        """
        1. Normal operation  
        2. Carpet pattern associated with possible lubrication issues  
        3. Structural looseness
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Monitoring uses the latest measurement for each equipment up to the selected date. "
        "Semantic AI Query uses approved SQL routes, keyword matching, embeddings, cosine similarity, and SQL Guard."
    )

    return selected_scenario_label, selected_collection_date


def filter_data(df: pd.DataFrame, selected_scenario_label: str) -> pd.DataFrame:
    """Filter diagnostic data according to selected scenario."""
    if df.empty:
        return df.copy()

    if selected_scenario_label == "All scenarios":
        return df.copy()

    if "scenario_label" not in df.columns:
        return df.copy()

    return df[df["scenario_label"] == selected_scenario_label].copy()


# -----------------------------------------------------------------------------
# Deterministic status and action-plan logic
# -----------------------------------------------------------------------------


def get_row_risk_score(row: pd.Series) -> float:
    """Return the strongest available risk signal for a row."""
    risk_fields = [
        "anomaly_probability",
        "max_anomaly_probability",
        "avg_anomaly_probability",
        "anomaly_score",
        "overall_anomaly_score",
        "max_anomaly_score",
        "avg_anomaly_score",
    ]

    values = [safe_float(row.get(field)) for field in risk_fields if field in row.index]
    return max(values) if values else 0.0


def classify_equipment_status(row: pd.Series) -> dict:
    """Classify equipment condition based on latest severity and risk indicators."""
    severity = str(row.get("severity", "")).lower()
    severity_level = str(row.get("severity_level", "")).lower()
    max_severity = str(row.get("max_recorded_severity", "")).lower()
    severity_reference = severity or severity_level or max_severity
    risk_score = get_row_risk_score(row)

    if severity_reference in {"critical", "high"} or risk_score >= 0.70:
        return {
            "status_label": "Critical",
            "status_emoji": "🔴",
            "status_color": "#D32F2F",
            "status_description": "Immediate attention required",
        }

    if severity_reference in {"attention", "medium"} or risk_score >= 0.40:
        return {
            "status_label": "Attention",
            "status_emoji": "🟡",
            "status_color": "#FBC02D",
            "status_description": "Schedule inspection and trend validation",
        }

    return {
        "status_label": "Normal",
        "status_emoji": "🟢",
        "status_color": "#388E3C",
        "status_description": "No immediate action required",
    }


def infer_action_plan(row: pd.Series) -> str:
    """Infer a deterministic recommended action from diagnostic features."""
    status = classify_equipment_status(row)["status_label"]

    existing_action = first_existing_value(row, ["recommended_action"], default="")
    if existing_action:
        return existing_action

    scenario_text = " ".join(
        str(row.get(field, ""))
        for field in [
            "scenario_name",
            "scenario_label",
            "predicted_condition",
            "predicted_label",
            "diagnostic_explanation",
            "explanation",
        ]
    ).lower()

    broadband_energy = safe_float(row.get("broadband_energy"))
    high_frequency_energy = safe_float(row.get("high_frequency_energy"))
    low_frequency_energy = safe_float(row.get("low_frequency_energy"))
    harmonic_ratio = safe_float(row.get("harmonic_ratio"))
    subharmonic_ratio = safe_float(row.get("subharmonic_ratio"))

    if status == "Normal" or "normal" in scenario_text:
        return (
            "Nenhuma ação requerida. Manter monitoramento periódico, acompanhar tendência "
            "de RMS velocity e registrar nova medição no próximo ciclo operacional."
        )

    if (
        "lubrication" in scenario_text
        or "lubr" in scenario_text
        or "carpet" in scenario_text
        or broadband_energy >= 0.50
        or high_frequency_energy >= 0.50
    ):
        return (
            "Priorizar inspeção de lubrificação: verificar nível e condição do lubrificante, "
            "contaminação, temperatura, condição de rolamentos e evolução de energia broadband/alta frequência."
        )

    if (
        "looseness" in scenario_text
        or "folga" in scenario_text
        or "holgura" in scenario_text
        or low_frequency_energy >= 0.50
        or harmonic_ratio >= 0.50
        or subharmonic_ratio >= 0.20
    ):
        return (
            "Priorizar inspeção mecânica estrutural: verificar base, fundação, parafusos, suportes, "
            "alinhamento, folgas mecânicas e componentes sujeitos a vibração em baixa frequência."
        )

    if status == "Critical":
        return (
            "Executar inspeção prioritária do ativo, validar diagnóstico com especialista de manutenção "
            "e planejar intervenção conforme criticidade operacional."
        )

    if status == "Attention":
        return (
            "Agendar inspeção técnica, comparar com histórico recente e acompanhar tendência de score, "
            "probabilidade de anomalia e velocidade RMS."
        )

    return "Nenhuma ação requerida. Manter monitoramento periódico."


# -----------------------------------------------------------------------------
# Card preparation and rendering
# -----------------------------------------------------------------------------


def get_latest_equipment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one latest diagnostic row per equipment."""
    if df.empty:
        return df.copy()

    latest_df = df.copy()

    group_column = "asset_id" if "asset_id" in latest_df.columns else "asset_name"

    if group_column not in latest_df.columns:
        return latest_df.reset_index(drop=True)

    if "timestamp" in latest_df.columns:
        latest_df["_timestamp_sort"] = pd.to_datetime(
            latest_df["timestamp"],
            errors="coerce",
        )
    else:
        latest_df["_timestamp_sort"] = pd.NaT

    if "measurement_id" in latest_df.columns:
        latest_df["_measurement_sort"] = pd.to_numeric(
            latest_df["measurement_id"],
            errors="coerce",
        )
    else:
        latest_df["_measurement_sort"] = 0

    latest_df = latest_df.sort_values(
        [group_column, "_timestamp_sort", "_measurement_sort"],
        ascending=[True, False, False],
    )

    latest_df = (
        latest_df.groupby(group_column, as_index=False)
        .head(1)
        .drop(columns=["_timestamp_sort", "_measurement_sort"], errors="ignore")
        .reset_index(drop=True)
    )

    return latest_df


def prepare_equipment_card_rows(df: pd.DataFrame, one_card_per_equipment: bool) -> pd.DataFrame:
    """Prepare rows for card rendering and enrich them with status/action plan."""
    if df.empty:
        return df.copy()

    prepared_df = normalize_card_columns(df)

    if one_card_per_equipment:
        prepared_df = get_latest_equipment_rows(prepared_df)
    else:
        prepared_df = prepared_df.reset_index(drop=True)

    status_values = prepared_df.apply(classify_equipment_status, axis=1)
    prepared_df["status_label"] = status_values.apply(lambda item: item["status_label"])
    prepared_df["status_emoji"] = status_values.apply(lambda item: item["status_emoji"])
    prepared_df["status_description"] = status_values.apply(
        lambda item: item["status_description"]
    )
    prepared_df["action_plan"] = prepared_df.apply(infer_action_plan, axis=1)
    prepared_df["risk_score"] = prepared_df.apply(get_row_risk_score, axis=1)

    status_order = {"Critical": 0, "Attention": 1, "Normal": 2}
    prepared_df["_status_order"] = prepared_df["status_label"].map(status_order).fillna(3)

    prepared_df = prepared_df.sort_values(
        ["_status_order", "risk_score"],
        ascending=[True, False],
    ).drop(columns=["_status_order"], errors="ignore")

    return prepared_df.reset_index(drop=True)


def infer_diagnostic_chart_type(row: pd.Series) -> str:
    """Infer which compact chart best explains the current diagnostic card."""
    scenario_text = " ".join(
        str(row.get(field, ""))
        for field in [
            "scenario_name",
            "scenario_label",
            "predicted_condition",
            "predicted_label",
            "diagnostic_explanation",
            "explanation",
        ]
    ).lower()

    broadband_energy = safe_float(row.get("broadband_energy"))
    high_frequency_energy = safe_float(row.get("high_frequency_energy"))
    low_frequency_energy = safe_float(row.get("low_frequency_energy"))
    harmonic_ratio = safe_float(row.get("harmonic_ratio"))
    subharmonic_ratio = safe_float(row.get("subharmonic_ratio"))

    if (
        "looseness" in scenario_text
        or "folga" in scenario_text
        or "holgura" in scenario_text
        or low_frequency_energy >= 0.50
        or harmonic_ratio >= 0.50
        or subharmonic_ratio >= 0.20
    ):
        return "structural_looseness"

    if (
        "lubrication" in scenario_text
        or "lubr" in scenario_text
        or "carpet" in scenario_text
        or broadband_energy >= 0.50
        or high_frequency_energy >= 0.50
    ):
        return "carpet_lubrication"

    return "general_condition"


def build_diagnostic_chart_dataframe(row: pd.Series) -> tuple[pd.DataFrame, str]:
    """Build compact chart data that supports the card diagnosis."""
    chart_type = infer_diagnostic_chart_type(row)

    if chart_type == "structural_looseness":
        chart_title = "Evidence: low-frequency looseness pattern"
        chart_data = pd.DataFrame(
            {
                "feature": [
                    "Low frequency",
                    "Harmonic ratio",
                    "Subharmonic ratio",
                ],
                "value": [
                    safe_float(row.get("low_frequency_energy")),
                    safe_float(row.get("harmonic_ratio")),
                    safe_float(row.get("subharmonic_ratio")),
                ],
            }
        )
        return chart_data, chart_title

    if chart_type == "carpet_lubrication":
        chart_title = "Evidence: carpet / lubrication pattern"
        chart_data = pd.DataFrame(
            {
                "feature": [
                    "Low frequency",
                    "Mid frequency",
                    "High frequency",
                    "Broadband",
                ],
                "value": [
                    safe_float(row.get("low_frequency_energy")),
                    safe_float(row.get("mid_frequency_energy")),
                    safe_float(row.get("high_frequency_energy")),
                    safe_float(row.get("broadband_energy")),
                ],
            }
        )
        return chart_data, chart_title

    chart_title = "Evidence: general condition indicators"
    chart_data = pd.DataFrame(
        {
            "feature": [
                "RMS velocity",
                "Peak velocity",
                "Anomaly score",
                "Anomaly probability",
            ],
            "value": [
                safe_float(row.get("rms_velocity")),
                safe_float(row.get("peak_velocity")),
                safe_float(row.get("anomaly_score")),
                safe_float(row.get("anomaly_probability")),
            ],
        }
    )
    return chart_data, chart_title


def render_diagnostic_evidence_chart(row: pd.Series, chart_key: str) -> None:
    """Render a compact evidence chart for a single equipment card."""
    chart_df, chart_title = build_diagnostic_chart_dataframe(row)

    if chart_df.empty:
        return

    if chart_df["value"].fillna(0).sum() == 0:
        return

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["feature"],
                y=chart_df["value"],
                text=chart_df["value"].round(3),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title={"text": chart_title, "font": {"size": 12}},
        height=210,
        margin={"l": 8, "r": 8, "t": 42, "b": 36},
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,
    )

    fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key=chart_key,
    )


def render_equipment_cards(
    result_df: pd.DataFrame,
    *,
    sql_query: str | None = None,
    one_card_per_equipment: bool = True,
    max_cards: int = 12,
    show_full_table: bool = True,
    show_raw_vibration_charts: bool = False,
) -> None:
    """Render equipment-oriented cards with status and deterministic action plan."""
    if result_df.empty:
        st.info("No rows returned.")

        if sql_query:
            with st.expander("SQL template", expanded=False):
                st.code(sql_query.strip(), language="sql")

        return

    card_df = prepare_equipment_card_rows(
        result_df,
        one_card_per_equipment=one_card_per_equipment,
    )

    cards_to_render = card_df.head(max_cards).reset_index(drop=True)
    cards_per_row = 3

    st.caption(f"Showing {len(cards_to_render)} equipment card(s).")

    for start_index in range(0, len(cards_to_render), cards_per_row):
        columns = st.columns(cards_per_row)

        for column_index, column_container in enumerate(columns):
            row_index = start_index + column_index

            if row_index >= len(cards_to_render):
                continue

            row = cards_to_render.iloc[row_index]
            status = classify_equipment_status(row)
            accent_color = status["status_color"]

            title_parts = []

            asset_code = first_existing_value(row, ["asset_code"], default="")
            asset_name = first_existing_value(row, ["asset_name"], default="")

            if asset_code and asset_name and asset_code != asset_name:
                title_parts.append(f"{asset_code} | {asset_name}")
            elif asset_name:
                title_parts.append(asset_name)
            elif asset_code:
                title_parts.append(asset_code)
            else:
                title_parts.append(f"Result {row_index + 1}")

            scenario_label = first_existing_value(
                row,
                ["scenario_label", "scenario_name"],
                default="",
            )
            if scenario_label:
                title_parts.append(scenario_label)

            if "measurement_id" in card_df.columns and pd.notna(row.get("measurement_id")):
                title_parts.append(f"Measurement {row['measurement_id']}")

            card_title = escape(" | ".join(title_parts))

            metric_items = [
                ("Status", "status_label"),
                ("Risk score", "risk_score"),
                ("Severity", "severity_level"),
                ("RMS velocity", "rms_velocity"),
                ("Peak velocity", "peak_velocity"),
                ("Anomaly score", "anomaly_score"),
                ("Anomaly probability", "anomaly_probability"),
                ("Carpet score", "carpet_score"),
                ("Looseness score", "looseness_score"),
                ("Avg anomaly score", "avg_anomaly_score"),
                ("Max anomaly score", "max_anomaly_score"),
                ("Avg anomaly probability", "avg_anomaly_probability"),
                ("Max anomaly probability", "max_anomaly_probability"),
                ("Total measurements", "total_measurements"),
            ]

            metric_html_parts = []

            for label, field in metric_items:
                if field in card_df.columns and pd.notna(row.get(field)):
                    value = row[field]

                    if isinstance(value, float):
                        value = f"{value:.3f}"

                    if field == "status_label":
                        value = f"{row.get('status_emoji', '')} {value}"

                    metric_html_parts.append(
                        "<div style='margin-bottom:8px;'>"
                        f"<div style='font-size:11px;color:#6B7280;line-height:1.1;'>{escape(label)}</div>"
                        f"<div style='font-size:16px;font-weight:700;color:#111827;line-height:1.2;'>{escape(str(value))}</div>"
                        "</div>"
                    )

            metric_html = "".join(metric_html_parts[:6])

            detail_fields = [
                ("Asset type", "asset_type"),
                ("Location", "location"),
                ("Scenario", "scenario_name"),
                ("Predicted condition", "predicted_condition"),
                ("Predicted label", "predicted_label"),
                ("Timestamp", "timestamp"),
                ("Sensor", "sensor_position"),
                ("RPM", "rpm"),
                ("Sampling rate", "sampling_rate_hz"),
            ]

            detail_html_parts = []

            for label, field in detail_fields:
                if field in card_df.columns and pd.notna(row.get(field)):
                    detail_html_parts.append(
                        "<div style='font-size:11.5px;line-height:1.35;margin-bottom:3px;color:#374151;'>"
                        f"<strong>{escape(label)}:</strong> {escape(str(row[field]))}"
                        "</div>"
                    )

            detail_html = "".join(detail_html_parts)

            explanation_value = first_existing_value(
                row,
                ["diagnostic_explanation", "explanation"],
                default="",
            )
            explanation_html = ""

            if explanation_value:
                explanation_html = (
                    "<div style='font-size:11.5px;line-height:1.35;margin-top:8px;color:#374151;'>"
                    "<strong>Diagnostic evidence:</strong><br>"
                    f"{escape(explanation_value)}"
                    "</div>"
                )

            action_plan = escape(str(row.get("action_plan", "Nenhuma ação requerida.")))

            card_html = (
                f"<div style='"
                f"border-left:7px solid {accent_color};"
                "border-top:1px solid #E5E7EB;"
                "border-right:1px solid #E5E7EB;"
                "border-bottom:1px solid #E5E7EB;"
                "border-radius:10px;"
                "padding:13px 14px 12px 14px;"
                "margin-bottom:14px;"
                "background-color:#FFFFFF;"
                "box-shadow:0 1px 3px rgba(0,0,0,0.05);"
                "min-height:360px;"
                "'>"
                "<div style='"
                "font-size:15px;"
                "font-weight:700;"
                "line-height:1.25;"
                "color:#111827;"
                "margin-bottom:12px;"
                f"'>{card_title}</div>"
                f"{metric_html}"
                "<div style='border-top:1px solid #F0F0F0;margin:10px 0;'></div>"
                f"{detail_html}"
                f"{explanation_html}"
                "<div style='font-size:11.5px;line-height:1.35;margin-top:10px;color:#111827;"
                "background-color:#F9FAFB;border-radius:8px;padding:8px;'>"
                "<strong>Recommended action plan:</strong><br>"
                f"{action_plan}"
                "</div>"
                "</div>"
            )

            with column_container:
                st.markdown(card_html, unsafe_allow_html=True)
                render_diagnostic_evidence_chart(
                    row,
                    chart_key=f"diagnostic_evidence_{start_index}_{column_index}_{row_index}",
                )

                if show_raw_vibration_charts:
                    with st.expander("Raw vibration analysis", expanded=False):
                        render_vibration_charts_for_measurement(
                            row,
                            chart_key_prefix=f"vibration_{start_index}_{column_index}_{row_index}",
                        )

    if len(card_df) > max_cards:
        st.info(
            f"{len(card_df) - max_cards} additional equipment row(s) are available in the full table below."
        )

    if sql_query:
        with st.expander("SQL template", expanded=False):
            st.code(sql_query.strip(), language="sql")

    if show_full_table:
        full_table_df = card_df.drop(columns=["risk_score"], errors="ignore")

        with st.expander("Show full result table", expanded=False):
            st.dataframe(
                full_table_df,
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------------------------------------------------------
# Monitoring mode
# -----------------------------------------------------------------------------


def render_monitoring_mode(
    monitoring_df: pd.DataFrame,
    *,
    selected_collection_date: str | None,
) -> None:
    """Render latest equipment monitoring cards."""
    st.header("Monitoring")
    st.markdown(
        """
        Latest available condition by equipment. Each card uses the most recent
        measurement for that equipment up to the selected collection date and
        applies a deterministic rule-based action plan. These actions are not generated by external AI services.
        """
    )

    if selected_collection_date:
        st.info(
            f"Monitoring snapshot: latest reading for each equipment up to **{selected_collection_date}**."
        )

    latest_equipment_df = prepare_equipment_card_rows(
        monitoring_df,
        one_card_per_equipment=True,
    )

    if latest_equipment_df.empty:
        st.warning("No monitoring data found for the selected filter/date.")
        return

    total_assets = (
        latest_equipment_df["asset_name"].nunique()
        if "asset_name" in latest_equipment_df.columns
        else len(latest_equipment_df)
    )
    critical_assets = int((latest_equipment_df["status_label"] == "Critical").sum())
    attention_assets = int((latest_equipment_df["status_label"] == "Attention").sum())
    normal_assets = int((latest_equipment_df["status_label"] == "Normal").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored equipment", total_assets)
    col2.metric("Critical", critical_assets)
    col3.metric("Attention", attention_assets)
    col4.metric("Normal", normal_assets)

    st.markdown("### Equipment status, action plan and vibration evidence")
    render_equipment_cards(
        latest_equipment_df,
        one_card_per_equipment=False,
        max_cards=24,
        show_full_table=True,
        show_raw_vibration_charts=True,
    )

    with st.expander("Monitoring trends", expanded=False):
        trend_df = load_diagnostic_data()
        if selected_collection_date and "collection_date" in trend_df.columns:
            trend_df = trend_df[
                trend_df["collection_date"].astype(str) <= str(selected_collection_date)
            ].copy()
        render_charts(trend_df)


# -----------------------------------------------------------------------------
# Semantic AI Query mode
# -----------------------------------------------------------------------------


def set_question_from_suggestion() -> None:
    """Copy selected approved suggestion into the free-text question field."""
    selected_label = st.session_state.get("selected_blocked_question_suggestion")
    suggestions = st.session_state.get("blocked_question_suggestions", [])

    if not selected_label or not suggestions:
        return

    suggestion_labels = [
        (
            f"{item['suggested_question']} "
            f"| intent: {item['intent']} "
            f"| score: {item['similarity_score']}"
        )
        for item in suggestions
    ]

    if selected_label not in suggestion_labels:
        return

    selected_index = suggestion_labels.index(selected_label)
    approved_question = suggestions[selected_index]["suggested_question"]

    st.session_state["ask_db_user_question"] = approved_question


def render_query_execution_summary(
    router_response: dict,
    validation_message: str,
    is_valid: bool,
    result_df: pd.DataFrame,
) -> None:
    """Render query routing, SQL safety, and execution-path diagnostics."""
    with st.expander("Query execution summary", expanded=False):
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.markdown("#### Semantic router")

            routing_method = router_response.get("routing_method", "unknown")

            if router_response["status"] == "matched":
                if routing_method == "keyword_match":
                    st.success("Matched by approved keyword template")
                elif routing_method == "embedding_similarity":
                    st.success("Matched by semantic similarity")
                else:
                    st.success("Template matched")

                st.caption(f"Intent: {router_response['intent']}")
                st.caption(f"Routing method: {routing_method}")

                if router_response.get("similarity_score") is not None:
                    st.caption(f"Similarity score: {router_response['similarity_score']}")

                if router_response.get("matched_example"):
                    st.caption(
                        f"Closest approved example: {router_response['matched_example']}"
                    )
            else:
                st.warning("No approved route found")
                st.caption("No keyword or embedding route matched the question.")

                if router_response.get("similarity_score") is not None:
                    st.caption(f"Best similarity score: {router_response['similarity_score']}")

                if router_response.get("matched_example"):
                    st.caption(
                        f"Closest approved example: {router_response['matched_example']}"
                    )

        with summary_col2:
            st.markdown("#### SQL safety")

            if is_valid:
                st.success(validation_message)
            else:
                st.error(validation_message)

            st.caption(f"Records returned: {len(result_df)}")

        with summary_col3:
            st.markdown("#### Execution path")

            if router_response["status"] == "matched":
                execution_path = {
                    "domain_guard": "passed",
                    "semantic_router": "matched",
                    "routing_method": router_response.get("routing_method", "unknown"),
                    "matched_intent": router_response["intent"],
                    "similarity_score": router_response.get("similarity_score"),
                    "matched_example": router_response.get("matched_example"),
                    "template_fallback": "not used",
                    "external_llm_usage": "not used",
                    "sql_guard": "passed" if is_valid else "blocked",
                }
            else:
                execution_path = {
                    "domain_guard": "passed",
                    "semantic_router": "not matched",
                    "routing_method": router_response.get("routing_method", "not_matched"),
                    "matched_intent": None,
                    "similarity_score": router_response.get("similarity_score"),
                    "matched_example": router_response.get("matched_example"),
                    "template_fallback": "used",
                    "external_llm_usage": "not used",
                    "sql_guard": "handled by controlled template fallback",
                }

            st.json(execution_path)


def render_blocked_query_summary(router_response: dict) -> None:
    """Render execution summary for blocked prompts."""
    st.markdown("### Execution summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.markdown("#### Question routing")
        st.error("Blocked")
        st.caption("Question outside supported maintenance domain.")

    with summary_col2:
        st.markdown("#### SQL validation")
        st.warning("SQL validation was not executed.")

    with summary_col3:
        st.markdown("#### Flow")
        st.json(
            {
                "domain_guard": "blocked",
                "semantic_router": "not executed for SQL routing",
                "clarification_suggestions": "not available",
                "matched_intent": None,
                "template_fallback": "not used",
                "external_llm_usage": "not used",
                "sql_guard": "not executed",
                "message": router_response.get("message"),
            }
        )


def render_chat_genai_mode() -> None:
    """Render the safe semantic database consultation interface."""
    st.header("Semantic AI Query")

    st.markdown(
        """
        ### Query predictive maintenance data using controlled semantic routing

        Ask operational questions about assets, vibration patterns, anomaly risk,
        lubrication issues, structural looseness, and cases requiring human validation.
        This version uses a zero-cost semantic layer: Domain Guard,
        keyword routing, embeddings, cosine similarity, approved SQL templates,
        and SQL Guard.
        """
    )

    hero_col1, hero_col2, hero_col3 = st.columns(3)

    with hero_col1:
        st.metric("Query mode", "Semantic AI")

    with hero_col2:
        st.metric("SQL execution", "Approved templates")

    with hero_col3:
        st.metric("Action plan", "Rule-based")

    st.info(
        "Main flow: user question → Domain Guard → Keyword Router → "
        "Embedding Router fallback → approved SQL template → SQL Guard → "
        "database results → deterministic action plan."
    )

    with st.expander("Example questions", expanded=False):
        supported_questions = get_supported_demo_questions()

        st.markdown(
            """
            Select one of the curated questions below or type a similar question.
            The examples are available in English, Portuguese, and Spanish.
            """
        )

        for question_group in supported_questions:
            st.markdown(f"**{question_group['category']}**")
            st.markdown(
                f"""
                - EN: `{question_group["english"]}`
                - PT-BR: `{question_group["portuguese"]}`
                - ES: `{question_group["spanish"]}`
                """
            )

    if "ask_db_user_question" not in st.session_state:
        st.session_state["ask_db_user_question"] = ""

    example_questions = get_supported_query_examples()

    st.markdown("---")

    st.markdown(
        """
        <div style="
            border:1px solid #D1D5DB;
            border-radius:14px;
            padding:18px 20px 10px 20px;
            margin-top:12px;
            margin-bottom:16px;
            background:linear-gradient(135deg, #F9FAFB 0%, #EEF2FF 100%);
            box-shadow:0 2px 6px rgba(0,0,0,0.04);
        ">
            <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:4px;">
                Ask with Semantic AI database routing
            </div>
            <div style="font-size:13px;color:#4B5563;margin-bottom:8px;">
                Start from an approved business question or write your own question in natural language.
                The system will route it to a safe SQL template before querying the database.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_example = st.selectbox(
        "Select an approved question to start",
        [""] + example_questions,
        key="ask_db_guided_question",
        help=(
            "Choose a curated question. The selected question will be copied "
            "to the free-text field and used as the input for the database query."
        ),
    )

    if selected_example and selected_example != st.session_state.get(
        "ask_db_last_selected_example"
    ):
        st.session_state["ask_db_user_question"] = selected_example
        st.session_state["ask_db_last_selected_example"] = selected_example

    if "blocked_question_suggestions" in st.session_state:
        st.markdown("#### Suggested approved questions")

        suggestions = st.session_state["blocked_question_suggestions"]

        st.info(
            "This question was not specific enough to run directly. "
            "Choose one approved question below or rewrite your question with more predictive maintenance context. "
            "The selected option will be copied to the free-text field."
        )

        suggestion_labels = [
            (
                f"{item['suggested_question']} "
                f"| intent: {item['intent']} "
                f"| score: {item['similarity_score']}"
            )
            for item in suggestions
        ]

        st.radio(
            "Choose an approved question",
            suggestion_labels,
            key="selected_blocked_question_suggestion",
            on_change=set_question_from_suggestion,
        )

        if st.button("Clear suggestions"):
            del st.session_state["blocked_question_suggestions"]
            st.rerun()

    user_question = st.text_input(
        "Question to run against the database",
        key="ask_db_user_question",
        placeholder="Example: qual equipamento tem maior risco e por qual motivo?",
        help=(
            "This is the final question used by the routing layer. "
            "You can select a guided question above or type directly here."
        ),
    )

    if st.button("Run Semantic AI database query", type="primary"):
        if not user_question.strip():
            st.warning("Please type or select a question.")
            return

        router_response = route_prompt_to_sql(user_question)

        if router_response["status"] == "blocked":
            suggestions = suggest_approved_questions(user_question, top_k=3)

            if suggestions:
                st.session_state["blocked_question_suggestions"] = suggestions
                st.rerun()

            st.error("This question is outside the supported predictive maintenance scope.")
            st.caption(router_response["message"])
            st.warning("No approved suggestions were found. Please rewrite your question.")
            render_blocked_query_summary(router_response)
            return

        if "blocked_question_suggestions" in st.session_state:
            del st.session_state["blocked_question_suggestions"]

        if router_response["status"] == "matched":
            sql_query = router_response["sql"]

            from src.database.queries import read_sql_query
            from src.llm.sql_guard import validate_sql_query

            is_valid, validation_message = validate_sql_query(sql_query)

            if is_valid:
                result_df = normalize_card_columns(read_sql_query(sql_query))
            else:
                result_df = pd.DataFrame()

        else:
            sql_query, result_df, validation_message = run_text_to_sql(user_question)
            result_df = normalize_card_columns(result_df)
            is_valid = (
                validation_message == "SQL query is valid."
                or validation_message == "SQL validation completed."
            )

        st.session_state["ask_db_last_result_df"] = result_df
        st.session_state["ask_db_last_sql_query"] = sql_query
        st.session_state["ask_db_last_router_response"] = router_response
        st.session_state["ask_db_last_validation_message"] = validation_message
        st.session_state["ask_db_last_is_valid"] = is_valid

    if "ask_db_last_result_df" in st.session_state:
        result_df = st.session_state["ask_db_last_result_df"]
        sql_query = st.session_state["ask_db_last_sql_query"]
        router_response = st.session_state["ask_db_last_router_response"]
        validation_message = st.session_state["ask_db_last_validation_message"]
        is_valid = st.session_state["ask_db_last_is_valid"]

        render_query_execution_summary(
            router_response=router_response,
            validation_message=validation_message,
            is_valid=is_valid,
            result_df=result_df,
        )

        with st.expander("Results", expanded=True):
            render_equipment_cards(
                result_df,
                sql_query=sql_query,
                one_card_per_equipment=True,
                max_cards=12,
                show_full_table=True,
                show_raw_vibration_charts=False,
            )


# -----------------------------------------------------------------------------
# Optional trend charts used inside Monitoring expander
# -----------------------------------------------------------------------------


def render_charts(df: pd.DataFrame) -> None:
    """Render exploratory vibration charts focused on timeline and spectral energy."""
    if df.empty:
        st.info("No data available for trend charts.")
        return

    df = normalize_card_columns(df)

    st.subheader("Vibration timeline")

    timeline_df = df.copy()

    if "timestamp" in timeline_df.columns:
        timeline_df["timestamp"] = pd.to_datetime(
            timeline_df["timestamp"],
            errors="coerce",
        )
        timeline_x = "timestamp"
    else:
        timeline_x = "measurement_id"

    y_column = "rms_velocity" if "rms_velocity" in timeline_df.columns else "rms_velocity_mm_s"

    if y_column not in timeline_df.columns:
        st.info("RMS velocity column is not available for the timeline chart.")
        return

    color_column = "scenario_label" if "scenario_label" in timeline_df.columns else None

    fig_timeline = px.line(
        timeline_df.sort_values(timeline_x),
        x=timeline_x,
        y=y_column,
        color=color_column,
        markers=True,
        hover_data=[
            column
            for column in [
                "measurement_id",
                "asset_name",
                "asset_type",
                "sensor_position",
                "peak_velocity",
                "anomaly_score",
                "anomaly_probability",
                "predicted_label",
                "predicted_condition",
            ]
            if column in timeline_df.columns
        ],
        title="RMS velocity timeline across monitored scenarios",
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Spectral energy profile")

    spectral_columns = [
        "low_frequency_energy",
        "mid_frequency_energy",
        "high_frequency_energy",
        "broadband_energy",
    ]

    available_spectral_columns = [
        column for column in spectral_columns if column in df.columns
    ]

    if available_spectral_columns and "scenario_label" in df.columns:
        spectral_profile_df = (
            df.groupby("scenario_label")[available_spectral_columns]
            .mean()
            .reset_index()
        )

        spectral_profile_long_df = spectral_profile_df.melt(
            id_vars="scenario_label",
            value_vars=available_spectral_columns,
            var_name="energy_band",
            value_name="average_energy",
        )

        fig_spectral = px.line(
            spectral_profile_long_df,
            x="energy_band",
            y="average_energy",
            color="scenario_label",
            markers=True,
            title="Average spectral energy profile by scenario",
        )

        st.plotly_chart(fig_spectral, use_container_width=True)
    else:
        st.info("Spectral energy columns are not available for this view.")


@st.cache_data(show_spinner=False)
def load_project_markdown() -> str:
    """Load the main project markdown used in the footer expander.

    Priority:
    1. docs/project_positioning.md, because it documents the current product narrative.
    2. README.md, as a fallback for environments where docs are not available.
    """

    candidate_paths = [
        PROJECT_ROOT / "docs" / "project_positioning.md",
        PROJECT_ROOT / "README.md",
    ]

    for markdown_path in candidate_paths:
        if markdown_path.exists():
            return markdown_path.read_text(encoding="utf-8")

    return (
        "Project documentation was not found. "
        "Expected one of: docs/project_positioning.md or README.md."
    )


def render_about_project_footer() -> None:
    """Render the project markdown in a footer expander."""

    st.markdown("---")

    with st.expander("About the Project", expanded=False):
        st.markdown(load_project_markdown())


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit app."""
    apply_sidebar_background()
    render_header()

    scenarios_df = load_scenarios()
    selected_scenario_label, selected_collection_date = render_sidebar(scenarios_df)

    app_mode = st.radio(
        "Choose the app mode",
        ["Monitoramento", "Semantic AI"],
        horizontal=True,
        key="app_mode_selector",
    )

    st.markdown("---")

    if app_mode == "Monitoramento":
        if not selected_collection_date:
            st.warning("No collection date is available for Monitoring.")
            render_about_project_footer()
            return

        monitoring_df = load_latest_measurements_until_date(selected_collection_date)
        filtered_monitoring_df = filter_data(monitoring_df, selected_scenario_label)
        render_monitoring_mode(
            filtered_monitoring_df,
            selected_collection_date=selected_collection_date,
        )
    elif app_mode == "Semantic AI":
        render_chat_genai_mode()

    render_about_project_footer()


if __name__ == "__main__":
    main()
