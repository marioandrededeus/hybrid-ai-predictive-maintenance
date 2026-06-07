"""
Streamlit app for the Hybrid AI Predictive Maintenance project.

This version focuses the user experience on two flows:
1. Monitoring: latest condition by equipment with rule-based action plans.
2. Chat GenAI: natural-language database consultation using controlled routing,
   approved SQL templates, embedding fallback, and SQL Guard.

No LLM or agent is used yet to generate action plans. The current action plan
layer is deterministic and rule-based.
"""

from __future__ import annotations

from pathlib import Path
from html import escape
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database.queries import get_full_diagnostic_view, get_scenarios  # noqa: E402
from src.llm.text_to_sql import run_text_to_sql  # noqa: E402
from src.llm.semantic_query_router import (  # noqa: E402
    get_supported_demo_questions,
    get_supported_query_examples,
    route_prompt_to_sql,
)
from src.llm.embedding_router import suggest_approved_questions  # noqa: E402


st.set_page_config(
    page_title="Hybrid AI for Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Data loading and global filters
# -----------------------------------------------------------------------------


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
        A practical lab for combining Machine Learning, controlled GenAI-style
        database interaction, SQL safety, and deterministic action planning in
        an industrial predictive maintenance scenario.
        """
    )


def render_sidebar(scenarios_df: pd.DataFrame) -> str:
    """Render sidebar filters and return selected scenario."""
    st.sidebar.header("Monitoring filter")

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

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "The monitoring view uses the latest available measurement for each equipment. "
        "The Chat GenAI view queries the database through approved SQL routes."
    )

    return selected_scenario_label


def filter_data(df: pd.DataFrame, selected_scenario_label: str) -> pd.DataFrame:
    """Filter diagnostic data according to selected scenario."""
    if selected_scenario_label == "All scenarios":
        return df.copy()

    return df[df["scenario_label"] == selected_scenario_label].copy()


# -----------------------------------------------------------------------------
# Deterministic status and action-plan logic
# -----------------------------------------------------------------------------


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float when possible."""
    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_row_risk_score(row: pd.Series) -> float:
    """Return the strongest available risk signal for a row."""
    risk_fields = [
        "anomaly_probability",
        "max_anomaly_probability",
        "avg_anomaly_probability",
        "anomaly_score",
        "max_anomaly_score",
        "avg_anomaly_score",
    ]

    values = [safe_float(row.get(field)) for field in risk_fields if field in row.index]
    return max(values) if values else 0.0


def classify_equipment_status(row: pd.Series) -> dict:
    """Classify equipment condition based on latest severity and risk indicators."""
    severity = str(row.get("severity_level", "")).lower()
    max_severity = str(row.get("max_recorded_severity", "")).lower()
    severity_reference = severity or max_severity
    risk_score = get_row_risk_score(row)

    if severity_reference == "high" or risk_score >= 0.70:
        return {
            "status_label": "Critical",
            "status_emoji": "🔴",
            "status_color": "#D32F2F",
            "status_description": "Immediate attention required",
        }

    if severity_reference == "medium" or risk_score >= 0.40:
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

    scenario_text = " ".join(
        str(row.get(field, ""))
        for field in ["scenario_name", "scenario_label", "predicted_label", "explanation"]
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


def get_criticality_color(row: pd.Series) -> str:
    """Return a card accent color based on deterministic status classification."""
    return classify_equipment_status(row)["status_color"]


# -----------------------------------------------------------------------------
# Card preparation and rendering
# -----------------------------------------------------------------------------


def get_latest_equipment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one latest diagnostic row per equipment."""
    if df.empty:
        return df.copy()

    latest_df = df.copy()

    if "asset_name" not in latest_df.columns:
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
        ["asset_name", "_timestamp_sort", "_measurement_sort"],
        ascending=[True, False, False],
    )

    latest_df = (
        latest_df.groupby("asset_name", as_index=False)
        .head(1)
        .drop(columns=["_timestamp_sort", "_measurement_sort"], errors="ignore")
        .reset_index(drop=True)
    )

    return latest_df


def prepare_equipment_card_rows(df: pd.DataFrame, one_card_per_equipment: bool) -> pd.DataFrame:
    """Prepare rows for card rendering and enrich them with status/action plan."""
    if df.empty:
        return df.copy()

    prepared_df = df.copy()

    if one_card_per_equipment and "asset_name" in prepared_df.columns:
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
        for field in ["scenario_name", "scenario_label", "predicted_label", "explanation"]
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

            if "asset_name" in card_df.columns and pd.notna(row.get("asset_name")):
                title_parts.append(str(row["asset_name"]))
            else:
                title_parts.append(f"Result {row_index + 1}")

            if "scenario_label" in card_df.columns and pd.notna(row.get("scenario_label")):
                title_parts.append(str(row["scenario_label"]))
            elif "scenario_name" in card_df.columns and pd.notna(row.get("scenario_name")):
                title_parts.append(str(row["scenario_name"]))

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
                ("Predicted label", "predicted_label"),
                ("Timestamp", "timestamp"),
                ("Sensor", "sensor_position"),
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

            explanation_html = ""

            if "explanation" in card_df.columns and pd.notna(row.get("explanation")):
                explanation_html = (
                    "<div style='font-size:11.5px;line-height:1.35;margin-top:8px;color:#374151;'>"
                    "<strong>Diagnostic evidence:</strong><br>"
                    f"{escape(str(row['explanation']))}"
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


def render_monitoring_mode(filtered_df: pd.DataFrame) -> None:
    """Render latest equipment monitoring cards."""
    st.header("Monitoring")
    st.markdown(
        """
        Latest available condition by equipment. Each card uses the most recent
        measurement for that equipment and applies a deterministic rule-based
        action plan. No LLM or agent is used to generate these actions yet.
        """
    )

    latest_equipment_df = prepare_equipment_card_rows(
        filtered_df,
        one_card_per_equipment=True,
    )

    if latest_equipment_df.empty:
        st.warning("No monitoring data found for the selected filter.")
        return

    total_assets = latest_equipment_df["asset_name"].nunique() if "asset_name" in latest_equipment_df.columns else len(latest_equipment_df)
    critical_assets = int((latest_equipment_df["status_label"] == "Critical").sum())
    attention_assets = int((latest_equipment_df["status_label"] == "Attention").sum())
    normal_assets = int((latest_equipment_df["status_label"] == "Normal").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored equipment", total_assets)
    col2.metric("Critical", critical_assets)
    col3.metric("Attention", attention_assets)
    col4.metric("Normal", normal_assets)

    st.markdown("### Equipment status and recommended action")
    render_equipment_cards(
        latest_equipment_df,
        one_card_per_equipment=False,
        max_cards=24,
        show_full_table=True,
    )

    with st.expander("Monitoring trends", expanded=False):
        render_charts(filtered_df)


# -----------------------------------------------------------------------------
# Chat GenAI mode
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
                    "text_to_sql_fallback": "not used",
                    "llm_usage": "not used yet",
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
                    "text_to_sql_fallback": "used",
                    "llm_usage": "not used yet",
                    "sql_guard": "handled by mock text-to-sql flow",
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
                "text_to_sql_fallback": "not used",
                "llm_usage": "not used yet",
                "sql_guard": "not executed",
                "message": router_response.get("message"),
            }
        )


def render_chat_genai_mode() -> None:
    """Render the safe natural-language database consultation interface."""
    st.header("Chat GenAI")

    st.markdown(
        """
        ### Query predictive maintenance data using natural language

        Ask operational questions about assets, vibration patterns, anomaly risk,
        lubrication issues, structural looseness, and cases requiring human validation.
        The current version uses controlled routing and SQL safety; LLM-based SQL
        generation will be added in a future step.
        """
    )

    hero_col1, hero_col2, hero_col3 = st.columns(3)

    with hero_col1:
        st.metric("Query mode", "Controlled GenAI")

    with hero_col2:
        st.metric("SQL generation", "Approved templates")

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
                Ask with GenAI-assisted database routing
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

    if st.button("Run GenAI-guided database query", type="primary"):
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
                result_df = read_sql_query(sql_query)
            else:
                result_df = pd.DataFrame()

        else:
            sql_query, result_df, validation_message = run_text_to_sql(user_question)
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
            )


# -----------------------------------------------------------------------------
# Optional trend charts used inside Monitoring expander
# -----------------------------------------------------------------------------


def render_charts(df: pd.DataFrame) -> None:
    """Render exploratory vibration charts focused on timeline and spectral energy."""
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

    fig_timeline = px.line(
        timeline_df.sort_values(timeline_x),
        x=timeline_x,
        y="rms_velocity",
        color="scenario_label",
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


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit app."""
    render_header()

    scenarios_df, diagnostic_df = load_data()

    selected_scenario_label = render_sidebar(scenarios_df)
    filtered_df = filter_data(diagnostic_df, selected_scenario_label)

    if filtered_df.empty:
        st.warning("No data found for the selected scenario.")
        return

    app_mode = st.radio(
        "Choose the app mode",
        ["Monitoramento", "Chat GenAI"],
        horizontal=True,
        key="app_mode_selector",
    )

    st.markdown("---")

    if app_mode == "Monitoramento":
        render_monitoring_mode(filtered_df)
    else:
        render_chat_genai_mode()


if __name__ == "__main__":
    main()
