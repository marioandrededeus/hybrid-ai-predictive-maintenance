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
from src.monitoring.human_feedback import (
    get_human_feedback_with_context,
    save_human_feedback,
)  # noqa: E402
from src.llm.text_to_sql import run_text_to_sql  # noqa: E402
from src.agents.maintenance_agent import run_maintenance_agent
from src.llm.semantic_query_router import (
    get_supported_query_examples,
    route_prompt_to_sql,
)

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

def render_human_validation(df: pd.DataFrame) -> None:
    """Render human validation form and feedback history."""
    st.subheader("Human Validation")

    st.markdown(
        """
        This section simulates the human-in-the-loop layer.
        A maintenance specialist can review a diagnostic result, validate the label,
        add notes, and register the required action.
        """
    )

    validation_options = df[
        [
            "measurement_id",
            "diagnostic_id",
            "asset_name",
            "scenario_label",
            "predicted_label",
            "anomaly_probability",
        ]
    ].copy()

    validation_options["option_label"] = (
        "Measurement "
        + validation_options["measurement_id"].astype(str)
        + " | "
        + validation_options["asset_name"]
        + " | "
        + validation_options["scenario_label"]
        + " | ML: "
        + validation_options["predicted_label"]
    )

    selected_option = st.selectbox(
        "Select a measurement to validate",
        validation_options["option_label"].tolist(),
    )

    selected_row = validation_options[
        validation_options["option_label"] == selected_option
    ].iloc[0]

    st.markdown("### Selected diagnostic")

    col1, col2, col3 = st.columns(3)

    col1.metric("Measurement ID", int(selected_row["measurement_id"]))
    col2.metric("Predicted label", selected_row["predicted_label"])
    col3.metric(
        "Anomaly probability",
        f"{selected_row['anomaly_probability']:.2f}",
    )

    validated_label = st.selectbox(
        "Validated label",
        [
            "normal_operation",
            "carpet_lubrication_issue",
            "structural_looseness",
            "inconclusive",
        ],
    )

    user_name = st.text_input(
        "Specialist name",
        value="Maintenance Specialist",
    )

    feedback_notes = st.text_area(
        "Feedback notes",
        placeholder="Add technical observations from the maintenance specialist.",
    )

    action_required = st.text_area(
        "Action required",
        placeholder="Describe the recommended or required maintenance action.",
    )

    if st.button("Save human validation"):
        save_human_feedback(
            measurement_id=int(selected_row["measurement_id"]),
            diagnostic_id=int(selected_row["diagnostic_id"]),
            user_name=user_name,
            validated_label=validated_label,
            feedback_notes=feedback_notes,
            action_required=action_required,
        )

        st.success("Human validation saved successfully.")

    st.markdown("---")
    st.markdown("### Human feedback history")

    feedback_df = get_human_feedback_with_context()

    if feedback_df.empty:
        st.info("No human feedback records found yet.")
    else:
        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True,
        )

def render_ask_database() -> None:
    """Render a safe database assistant with semantic routing."""
    st.subheader("Ask the Database")

    st.markdown(
        """
        Ask questions about the predictive maintenance database.
        The app first tries to match the question with a safe predefined SQL template.
        If no template is found, it falls back to the controlled mock Text-to-SQL layer.
        Only safe SELECT queries are allowed.
        """
    )

    example_questions = get_supported_query_examples()

    selected_example = st.selectbox(
        "Choose an example question",
        [""] + example_questions,
    )

    user_question = st.text_input(
        "Or type your own question",
        value=selected_example,
        placeholder="Example: Show average anomaly score by scenario",
    )

    if st.button("Run database question"):
        if not user_question.strip():
            st.warning("Please type or select a question.")
            return

        router_response = route_prompt_to_sql(user_question)

        if router_response["status"] == "blocked":
            st.error(router_response["message"])

            st.markdown("### Execution path")
            st.json(
                {
                    "domain_guard": "blocked",
                    "semantic_router": "not executed",
                    "matched_intent": None,
                    "text_to_sql_fallback": "not used",
                    "llm_usage": "not used",
                    "sql_guard": "not executed",
                }
            )

            return
        
        if router_response["status"] == "matched":
            sql_query = router_response["sql"]

            st.markdown("### Semantic router")
            st.success("Prompt matched a predefined SQL template.")
            st.info(f"Matched intent: {router_response['intent']}")

            st.markdown("### Execution path")
            st.json(
                {
                    "domain_guard": "passed",
                    "semantic_router": "matched",
                    "matched_intent": router_response["intent"],
                    "text_to_sql_fallback": "not used",
                    "llm_usage": "not used",
                    "sql_guard": "pending",
                }
            )

            from src.database.queries import read_sql_query
            from src.llm.sql_guard import validate_sql_query

            is_valid, validation_message = validate_sql_query(sql_query)

            if not is_valid:
                result_df = pd.DataFrame()
            else:
                result_df = read_sql_query(sql_query)

            st.caption(f"Rows returned: {len(result_df)}")

            if is_valid:
                st.success("SQL Guard: passed")
            else:
                st.error("SQL Guard: blocked")

            st.caption(f"Rows returned: {len(result_df)}")

        else:
            st.markdown("### Semantic router")
            st.warning(router_response["message"])
            st.info("Fallback: using controlled mock Text-to-SQL.")

            st.markdown("### Execution path")
            st.json(
                {
                    "domain_guard": "passed",
                    "semantic_router": "not matched",
                    "matched_intent": None,
                    "text_to_sql_fallback": "used",
                    "llm_usage": "not used",
                    "sql_guard": "handled by mock text-to-sql flow",
                }
            )

            sql_query, result_df, validation_message = run_text_to_sql(user_question)

        st.markdown("### SQL validation")
        if validation_message == "SQL query is valid." or validation_message == "SQL validation completed.":
            st.success(validation_message)
        else:
            st.error(validation_message)

        st.markdown("### Generated SQL")
        st.code(sql_query.strip(), language="sql")

        st.markdown("### Result")

        if result_df.empty:
            st.info("No rows returned.")
        else:
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

def main() -> None:
    """Run the Streamlit app."""
    render_header()

    scenarios_df, diagnostic_df = load_data()

    selected_scenario_label = render_sidebar(scenarios_df)
    filtered_df = filter_data(diagnostic_df, selected_scenario_label)

    if filtered_df.empty:
        st.warning("No data found for the selected scenario.")
        return

    tab_overview, tab_summary, tab_recommendations, tab_validation, tab_ask_db, tab_agent, tab_data = st.tabs(
        [
            "Overview",
            "Scenario Summary",
            "Recommendations",
            "Human Validation",
            "Ask the Database",
            "Maintenance Agent",
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

    with tab_validation:
        render_human_validation(filtered_df)

    with tab_ask_db:
        render_ask_database()

    with tab_data:
        render_data_table(filtered_df)

    with tab_agent:
        st.header("Maintenance Agent")

        st.write(
            "This deterministic agent orchestrates database queries, scenario summaries, "
            "rule-based recommendations, and operational reasoning without using a real LLM yet."
        )

        scenarios_df = get_scenarios()

        scenario_options = dict(
            zip(
                scenarios_df["scenario_label"],
                scenarios_df["scenario_id"]
            )
        )

        selected_scenario_label = st.selectbox(
            "Select an industrial scenario",
            options=list(scenario_options.keys()),
            key="agent_scenario_selector"
        )

        selected_scenario_id = scenario_options[selected_scenario_label]

        if st.button("Run Maintenance Agent"):
            agent_response = run_maintenance_agent(selected_scenario_id)

            st.subheader("Agent Status")

            if agent_response["status"] == "success":
                st.success(agent_response["message"])
            else:
                st.error(agent_response["message"])

            st.subheader("Selected Scenario")
            st.json(agent_response["scenario"])

            st.subheader("Scenario Summary")

            if agent_response["summary"] is not None:
                st.json(agent_response["summary"])
            else:
                st.warning("No summary found for this scenario.")

            st.subheader("Recommendations")
            recommendations_df = pd.DataFrame(agent_response["recommendations"])

            st.dataframe(
                recommendations_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Agent Reasoning")
            for step in agent_response["agent_reasoning"]:
                st.markdown(f"- {step}")
    


if __name__ == "__main__":
    main()