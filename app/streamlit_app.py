"""
Streamlit app for the Hybrid AI Predictive Maintenance project.

This first version reads pre-populated SQLite data and allows the user
to explore industrial vibration scenarios.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import get_full_diagnostic_view, get_scenarios  # noqa: E402
from src.analysis.scenario_summary import generate_scenario_summaries, generate_text_summary  # noqa: E402
from src.prescription.rule_based_recommendations import generate_recommendations  # noqa: E402

st.set_page_config(
    page_title="Hybrid AI for Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load scenarios and diagnostic data from SQLite."""
    scenarios_df = get_scenarios()
    diagnostic_df = get_full_diagnostic_view()

    return scenarios_df, diagnostic_df


def render_header() -> None:
    """Render the app header."""
    st.title("Hybrid AI for Predictive Maintenance")

    st.markdown(
        """
        Practical and didactic lab for exploring the integration between
        Machine Learning, LLMs, agents, RAG, and human validation in an
        industrial predictive maintenance scenario.
        """
    )


def render_sidebar(scenarios_df: pd.DataFrame) -> str:
    """Render sidebar filters and return selected scenario."""
    st.sidebar.header("Scenario selection")

    scenario_options = ["All scenarios"] + scenarios_df["scenario_label"].tolist()

    selected_scenario_label = st.sidebar.selectbox(
        "Choose an industrial scenario",
        scenario_options,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Initial scenarios")
    st.sidebar.markdown(
        """
        1. Normal operation  
        2. Carpet pattern associated with possible lubrication issues  
        3. Structural looseness
        """
    )

    return selected_scenario_label


def filter_data(df: pd.DataFrame, selected_scenario_label: str) -> pd.DataFrame:
    """Filter data according to selected scenario."""
    if selected_scenario_label == "All scenarios":
        return df

    return df[df["scenario_label"] == selected_scenario_label].copy()


def render_kpis(df: pd.DataFrame) -> None:
    """Render basic KPI cards."""
    total_measurements = len(df)
    avg_anomaly_score = df["anomaly_score"].mean()
    avg_rms_velocity = df["rms_velocity"].mean()
    max_anomaly_probability = df["anomaly_probability"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Measurements", total_measurements)
    col2.metric("Avg anomaly score", f"{avg_anomaly_score:.2f}")
    col3.metric("Avg RMS velocity", f"{avg_rms_velocity:.2f}")
    col4.metric("Max anomaly probability", f"{max_anomaly_probability:.2f}")


def render_charts(df: pd.DataFrame) -> None:
    """Render exploratory charts."""
    st.subheader("Anomaly score by measurement")

    fig = px.bar(
        df,
        x="measurement_id",
        y="anomaly_score",
        color="scenario_label",
        hover_data=[
            "asset_name",
            "asset_type",
            "sensor_position",
            "rms_velocity",
            "peak_velocity",
            "broadband_energy",
            "predicted_label",
        ],
        title="Anomaly score across industrial scenarios",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spectral feature overview")

    spectral_columns = [
        "low_frequency_energy",
        "mid_frequency_energy",
        "high_frequency_energy",
        "broadband_energy",
        "harmonic_ratio",
        "subharmonic_ratio",
    ]

    selected_feature = st.selectbox(
        "Choose a spectral feature",
        spectral_columns,
    )

    fig_feature = px.bar(
        df,
        x="measurement_id",
        y=selected_feature,
        color="scenario_label",
        hover_data=["asset_name", "scenario_label", "predicted_label"],
        title=f"{selected_feature} by measurement",
    )

    st.plotly_chart(fig_feature, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    """Render diagnostic table."""
    st.subheader("Diagnostic data")

    columns_to_show = [
        "measurement_id",
        "timestamp",
        "asset_name",
        "asset_type",
        "location",
        "scenario_label",
        "severity_level",
        "sensor_position",
        "rms_velocity",
        "peak_velocity",
        "crest_factor",
        "temperature_celsius",
        "load_percentage",
        "anomaly_score",
        "predicted_label",
        "anomaly_probability",
        "explanation",
    ]

    st.dataframe(
        df[columns_to_show],
        use_container_width=True,
        hide_index=True,
    )


def render_scenario_explanation(df: pd.DataFrame) -> None:
    """Render a simple explanation block based on the selected data."""
    st.subheader("Scenario interpretation")

    selected_labels = df["scenario_label"].unique().tolist()

    for label in selected_labels:
        scenario_data = df[df["scenario_label"] == label]
        explanations = scenario_data["explanation"].dropna().unique().tolist()

        with st.expander(label, expanded=True):
            for explanation in explanations:
                st.markdown(f"- {explanation}")

def render_scenario_summary(df: pd.DataFrame) -> None:
    """Render technical scenario summaries."""
    st.subheader("Scenario Summary")

    summaries_df = generate_scenario_summaries(df)

    st.dataframe(
        summaries_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Technical interpretation")

    for _, row in summaries_df.iterrows():
        with st.expander(row["scenario_label"], expanded=True):
            st.text(generate_text_summary(row))


def render_recommendations(df: pd.DataFrame) -> None:
    """Render rule-based maintenance recommendations."""
    st.subheader("Recommendations")

    recommendations_df = generate_recommendations(df)

    st.markdown(
        """
        These recommendations are generated using transparent rule-based logic.
        Later, this layer will be combined with RAG, LLM reasoning, agents,
        and human validation.
        """
    )

    priority_filter = st.selectbox(
        "Filter by priority",
        ["All priorities", "Low", "Medium", "High"],
    )

    if priority_filter != "All priorities":
        recommendations_df = recommendations_df[
            recommendations_df["priority"] == priority_filter
        ].copy()

    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Human validation queue")

    validation_df = recommendations_df[
        recommendations_df["human_validation_required"] == True
    ].copy()

    if validation_df.empty:
        st.success("No recommendations require human validation for the selected view.")
    else:
        st.warning(
            f"{len(validation_df)} recommendation(s) require human validation."
        )

        for _, row in validation_df.iterrows():
            with st.expander(
                f"Measurement {row['measurement_id']} | {row['priority']} priority",
                expanded=False,
            ):
                st.markdown(f"**Scenario:** {row['scenario_label']}")
                st.markdown(f"**Recommended action:** {row['recommended_action']}")
                st.markdown(f"**Technical reason:** {row['technical_reason']}")

def main() -> None:
    """Run the Streamlit app."""
    render_header()

    scenarios_df, diagnostic_df = load_data()

    selected_scenario_label = render_sidebar(scenarios_df)
    filtered_df = filter_data(diagnostic_df, selected_scenario_label)

    if filtered_df.empty:
        st.warning("No data found for the selected scenario.")
        return

    tab_overview, tab_summary, tab_recommendations, tab_data = st.tabs(
        [
            "Overview",
            "Scenario Summary",
            "Recommendations",
            "Diagnostic Data",
        ]
    )

    with tab_overview:
        render_kpis(filtered_df)
        render_charts(filtered_df)
        render_scenario_explanation(filtered_df)

    with tab_summary:
        render_scenario_summary(filtered_df)

    with tab_recommendations:
        render_recommendations(filtered_df)

    with tab_data:
        render_data_table(filtered_df)


if __name__ == "__main__":
    main()