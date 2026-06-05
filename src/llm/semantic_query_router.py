"""
Semantic query router for the Hybrid AI Predictive Maintenance project.

This module maps user questions to predefined safe SQL templates.
It acts as a low-cost deterministic layer before any future LLM-based
Text-to-SQL fallback.
"""

from src.llm.domain_guard import get_domain_guard_response


QUERY_TEMPLATES = [
    {
        "intent": "average_anomaly_score_by_scenario",
        "keywords": [
            "average anomaly score",
            "avg anomaly score",
            "anomaly score by scenario",
            "mean anomaly score",
        ],
        "sql": """
            SELECT
                s.scenario_label,
                ROUND(AVG(sf.anomaly_score), 3) AS avg_anomaly_score
            FROM spectral_features sf
            JOIN vibration_measurements vm
                ON sf.measurement_id = vm.measurement_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            GROUP BY s.scenario_label
            ORDER BY avg_anomaly_score DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "average_anomaly_probability_by_scenario",
        "keywords": [
            "average anomaly probability",
            "avg anomaly probability",
            "anomaly probability by scenario",
            "mean anomaly probability",
        ],
        "sql": """
            SELECT
                s.scenario_label,
                ROUND(AVG(md.anomaly_probability), 3) AS avg_anomaly_probability
            FROM ml_diagnostics md
            JOIN vibration_measurements vm
                ON md.measurement_id = vm.measurement_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            GROUP BY s.scenario_label
            ORDER BY avg_anomaly_probability DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "highest_risk_scenarios",
        "keywords": [
            "highest risk",
            "high risk",
            "risk level",
            "riskiest scenario",
            "which scenario has the highest risk",
        ],
        "sql": """
            SELECT
                s.scenario_label,
                s.severity_level,
                ROUND(AVG(sf.anomaly_score), 3) AS avg_anomaly_score,
                ROUND(AVG(md.anomaly_probability), 3) AS avg_anomaly_probability
            FROM scenarios s
            JOIN vibration_measurements vm
                ON s.scenario_id = vm.scenario_id
            LEFT JOIN spectral_features sf
                ON vm.measurement_id = sf.measurement_id
            LEFT JOIN ml_diagnostics md
                ON vm.measurement_id = md.measurement_id
            GROUP BY
                s.scenario_label,
                s.severity_level
            ORDER BY avg_anomaly_score DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "measurements_requiring_human_validation",
        "keywords": [
            "human validation",
            "require validation",
            "requires validation",
            "human review",
            "manual validation",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                s.scenario_label,
                s.severity_level,
                ROUND(sf.anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability
            FROM vibration_measurements vm
            JOIN assets a
                ON vm.asset_id = a.asset_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            LEFT JOIN spectral_features sf
                ON vm.measurement_id = sf.measurement_id
            LEFT JOIN ml_diagnostics md
                ON vm.measurement_id = md.measurement_id
            WHERE
                sf.anomaly_score >= 0.70
                OR md.anomaly_probability >= 0.70
                OR s.severity_level IN ('medium', 'high')
            ORDER BY
                anomaly_score DESC,
                anomaly_probability DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "average_rms_velocity_by_scenario",
        "keywords": [
            "average rms",
            "avg rms",
            "rms by scenario",
            "rms velocity by scenario",
            "average rms velocity",
        ],
        "sql": """
            SELECT
                s.scenario_label,
                ROUND(AVG(vm.rms_velocity), 3) AS avg_rms_velocity
            FROM vibration_measurements vm
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            GROUP BY s.scenario_label
            ORDER BY avg_rms_velocity DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "lubrication_issues",
        "keywords": [
            "lubrication issues",
            "lubrication problem",
            "lubrication degradation",
            "carpet pattern",
            "carpet",
            "high frequency energy",
            "broadband energy",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                a.asset_type,
                a.location,
                s.scenario_label,
                s.severity_level,
                ROUND(vm.rms_velocity, 3) AS rms_velocity,
                ROUND(vm.peak_velocity, 3) AS peak_velocity,
                ROUND(sf.broadband_energy, 3) AS broadband_energy,
                ROUND(sf.high_frequency_energy, 3) AS high_frequency_energy,
                ROUND(sf.anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_label,
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
            WHERE
                s.scenario_name = 'carpet_lubrication_issue'
                OR s.scenario_label LIKE '%lubrication%'
                OR s.scenario_label LIKE '%Carpet%'
            ORDER BY
                sf.anomaly_score DESC,
                sf.high_frequency_energy DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "structural_looseness_cases",
        "keywords": [
            "structural looseness",
            "looseness cases",
            "looseness",
            "low frequency energy",
            "harmonic ratio",
            "subharmonic ratio",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                a.asset_type,
                a.location,
                s.scenario_label,
                s.severity_level,
                ROUND(vm.rms_velocity, 3) AS rms_velocity,
                ROUND(vm.peak_velocity, 3) AS peak_velocity,
                ROUND(sf.low_frequency_energy, 3) AS low_frequency_energy,
                ROUND(sf.harmonic_ratio, 3) AS harmonic_ratio,
                ROUND(sf.subharmonic_ratio, 3) AS subharmonic_ratio,
                ROUND(sf.anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_label,
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
            WHERE
                s.scenario_name = 'structural_looseness'
                OR s.scenario_label LIKE '%looseness%'
            ORDER BY
                sf.anomaly_score DESC,
                sf.low_frequency_energy DESC,
                sf.harmonic_ratio DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "monitored_assets",
        "keywords": [
            "which assets are monitored",
            "monitored assets",
            "show assets",
            "list assets",
            "industrial assets",
            "which equipment",
            "monitored equipment",
        ],
        "sql": """
            SELECT
                asset_id,
                asset_name,
                asset_type,
                location,
                manufacturer,
                installation_year
            FROM assets
            ORDER BY asset_id
            LIMIT 50;
        """,
    },
    {
        "intent": "anomaly_risk_by_measurement",
        "keywords": [
            "anomaly risk by measurement",
            "risk by measurement",
            "anomaly by measurement",
            "show anomaly risk",
            "measurement risk",
            "risk per measurement",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                s.scenario_label,
                s.severity_level,
                ROUND(vm.rms_velocity, 3) AS rms_velocity,
                ROUND(vm.peak_velocity, 3) AS peak_velocity,
                ROUND(sf.anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_label,
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
            ORDER BY
                sf.anomaly_score DESC,
                md.anomaly_probability DESC
            LIMIT 50;
        """,
    },
]


def normalize_prompt(prompt: str) -> str:
    """
    Normalize user prompt before routing.
    """

    return prompt.strip().lower()


def route_prompt_to_sql(prompt: str) -> dict:
    """
    Route a user prompt to a predefined SQL template when possible.
    """

    domain_validation = get_domain_guard_response(prompt)

    if not domain_validation["is_allowed"]:
        return {
            "status": "blocked",
            "intent": None,
            "sql": None,
            "message": domain_validation["message"],
        }

    normalized_prompt = normalize_prompt(prompt)

    for template in QUERY_TEMPLATES:
        if any(keyword in normalized_prompt for keyword in template["keywords"]):
            return {
                "status": "matched",
                "intent": template["intent"],
                "sql": template["sql"],
                "message": "Prompt matched a predefined SQL template.",
            }

    return {
        "status": "not_matched",
        "intent": None,
        "sql": None,
        "message": (
            "Prompt is inside the predictive maintenance domain, "
            "but no predefined SQL template was found yet."
        ),
    }

def get_supported_query_examples() -> list[str]:
    """
    Return representative example questions supported by the semantic router.
    """

    examples = []

    for template in QUERY_TEMPLATES:
        if template["keywords"]:
            examples.append(template["keywords"][0])

    return examples