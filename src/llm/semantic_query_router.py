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
            "show average anomaly score by scenario",
            "média do score de anomalia",
            "media do score de anomalia",
            "score médio de anomalia",
            "score medio de anomalia",
            "score de anomalia por cenário",
            "score de anomalia por cenario",
            "média de anomalia por cenário",
            "media de anomalia por cenario",
            "promedio del score de anomalía",
            "promedio del score de anomalia",
            "score promedio de anomalía",
            "score promedio de anomalia",
            "score de anomalía por escenario",
            "score de anomalia por escenario",
            "promedio de anomalía por escenario",
            "promedio de anomalia por escenario",
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
            "highest risk scenarios",
            "show highest risk scenarios",
            "maior risco",
            "alto risco",
            "nível de risco",
            "nivel de risco",
            "cenário de maior risco",
            "cenario de maior risco",
            "cenários de maior risco",
            "cenarios de maior risco",
            "qual cenário tem maior risco",
            "qual cenario tem maior risco",
            "mostrar cenários de maior risco",
            "mostrar cenarios de maior risco",
            "mayor riesgo",
            "alto riesgo",
            "nivel de riesgo",
            "escenario de mayor riesgo",
            "escenarios de mayor riesgo",
            "qué escenario tiene mayor riesgo",
            "que escenario tiene mayor riesgo",
            "mostrar escenarios de mayor riesgo",
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
            "measurements requiring human validation",
            "show measurements requiring human validation",
            "validação humana",
            "validacao humana",
            "requer validação",
            "requer validacao",
            "requerem validação",
            "requerem validacao",
            "revisão humana",
            "revisao humana",
            "validação manual",
            "validacao manual",
            "medições que requerem validação humana",
            "medicoes que requerem validacao humana",
            "medidas que requerem validação humana",
            "medidas que requerem validacao humana",
            "validación humana",
            "validacion humana",
            "requiere validación",
            "requiere validacion",
            "requieren validación",
            "requieren validacion",
            "revisión humana",
            "revision humana",
            "validación manual",
            "validacion manual",
            "mediciones que requieren validación humana",
            "mediciones que requieren validacion humana",
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
            "show average rms velocity by scenario",
            "média de rms",
            "media de rms",
            "rms médio",
            "rms medio",
            "rms por cenário",
            "rms por cenario",
            "velocidade rms por cenário",
            "velocidade rms por cenario",
            "média da velocidade rms",
            "media da velocidade rms",
            "promedio de rms",
            "rms promedio",
            "rms por escenario",
            "velocidad rms por escenario",
            "promedio de velocidad rms",
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
            "lubrication failure",
            "problemas de lubrificação",
            "problemas de lubrificacao",
            "problema de lubrificação",
            "problema de lubrificacao",
            "degradação da lubrificação",
            "degradacao da lubrificacao",
            "falha de lubrificação",
            "falha de lubrificacao",
            "padrão carpet",
            "padrao carpet",
            "energia de alta frequência",
            "energia de alta frequencia",
            "energia broadband",
            "energia de banda larga",
            "problemas de lubricación",
            "problemas de lubricacion",
            "problema de lubricación",
            "problema de lubricacion",
            "degradación de lubricación",
            "degradacion de lubricacion",
            "falla de lubricación",
            "falla de lubricacion",
            "patrón carpet",
            "patron carpet",
            "energía de alta frecuencia",
            "energia de alta frecuencia",
            "energía broadband",
            "energia broadband",
            "energía de banda ancha",
            "energia de banda ancha",
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
            "structural looseness cases",
            "folga estrutural",
            "casos de folga estrutural",
            "folga",
            "energia de baixa frequência",
            "energia de baixa frequencia",
            "razão harmônica",
            "razao harmonica",
            "relação harmônica",
            "relacao harmonica",
            "razão sub-harmônica",
            "razao sub-harmonica",
            "relação sub-harmônica",
            "relacao sub-harmonica",
            "holgura estructural",
            "casos de holgura estructural",
            "holgura",
            "energía de baja frecuencia",
            "energia de baja frecuencia",
            "relación armónica",
            "relacion armonica",
            "ratio armónico",
            "ratio armonico",
            "relación subarmónica",
            "relacion subarmonica",
            "ratio subarmónico",
            "ratio subarmonico",
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
        "intent": "high_severity_diagnostics",
                "keywords": [
            "high severity diagnostics",
            "high severity cases",
            "high severity measurements",
            "high severity predictive maintenance diagnostics",
            "high severity predictive maintenance",
            "high severity vibration measurements",
            "critical diagnostic cases",
            "critical diagnostics",
            "critical maintenance cases",
            "critical vibration cases",
            "critical cases",
            "which cases are high severity",
            "show high severity diagnostics",
            "show critical diagnostics",
            "diagnósticos de alta severidade",
            "diagnosticos de alta severidade",
            "casos de alta severidade",
            "medições de alta severidade",
            "medicoes de alta severidade",
            "diagnósticos críticos",
            "diagnosticos criticos",
            "casos críticos",
            "casos criticos",
            "casos críticos de manutenção",
            "casos criticos de manutencao",
            "casos críticos de vibração",
            "casos criticos de vibracao",
            "quais casos são de alta severidade",
            "quais casos sao de alta severidade",
            "mostrar diagnósticos de alta severidade",
            "mostrar diagnosticos de alta severidade",
            "mostrar diagnósticos críticos",
            "mostrar diagnosticos criticos",
            "diagnósticos de alta severidad",
            "diagnosticos de alta severidad",
            "casos de alta severidad",
            "mediciones de alta severidad",
            "diagnósticos críticos",
            "diagnosticos criticos",
            "casos críticos",
            "casos criticos",
            "casos críticos de mantenimiento",
            "casos criticos de mantenimiento",
            "casos críticos de vibración",
            "casos criticos de vibracion",
            "qué casos son de alta severidad",
            "que casos son de alta severidad",
            "mostrar diagnósticos de alta severidad",
            "mostrar diagnosticos de alta severidad",
            "mostrar diagnósticos críticos",
            "mostrar diagnosticos criticos",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                a.asset_type,
                a.location,
                s.scenario_name,
                s.scenario_label,
                s.severity_level,
                vm.sensor_position,
                ROUND(vm.rms_velocity, 3) AS rms_velocity,
                ROUND(vm.peak_velocity, 3) AS peak_velocity,
                ROUND(vm.crest_factor, 3) AS crest_factor,
                ROUND(vm.temperature_celsius, 3) AS temperature_celsius,
                ROUND(vm.load_percentage, 3) AS load_percentage,
                ROUND(sf.dominant_frequency_hz, 3) AS dominant_frequency_hz,
                ROUND(sf.low_frequency_energy, 3) AS low_frequency_energy,
                ROUND(sf.mid_frequency_energy, 3) AS mid_frequency_energy,
                ROUND(sf.high_frequency_energy, 3) AS high_frequency_energy,
                ROUND(sf.broadband_energy, 3) AS broadband_energy,
                ROUND(sf.harmonic_ratio, 3) AS harmonic_ratio,
                ROUND(sf.subharmonic_ratio, 3) AS subharmonic_ratio,
                ROUND(sf.anomaly_score, 3) AS anomaly_score,
                md.predicted_label,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.model_name,
                md.model_version,
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
            WHERE LOWER(s.severity_level) = 'high'
            ORDER BY
                md.anomaly_probability DESC,
                sf.anomaly_score DESC,
                vm.timestamp
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
            "quais ativos são monitorados",
            "quais ativos sao monitorados",
            "ativos monitorados",
            "mostrar ativos",
            "listar ativos",
            "ativos industriais",
            "quais equipamentos",
            "equipamentos monitorados",
            "qué activos son monitoreados",
            "que activos son monitoreados",
            "activos monitoreados",
            "mostrar activos",
            "listar activos",
            "activos industriales",
            "qué equipos",
            "que equipos",
            "equipos monitoreados",
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
            "risco de anomalia por medição",
            "risco de anomalia por medicao",
            "risco por medição",
            "risco por medicao",
            "anomalia por medição",
            "anomalia por medicao",
            "mostrar risco de anomalia",
            "risco da medição",
            "risco da medicao",
            "risco por medida",
            "riesgo de anomalía por medición",
            "riesgo de anomalia por medicion",
            "riesgo por medición",
            "riesgo por medicion",
            "anomalía por medición",
            "anomalia por medicion",
            "mostrar riesgo de anomalía",
            "mostrar riesgo de anomalia",
            "riesgo de la medición",
            "riesgo de la medicion",
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
    {
        "intent": "highest_risk_assets",
        "keywords": [
            "highest risk assets",
            "assets with highest anomaly risk",
            "which assets have the highest anomaly risk",
            "highest anomaly risk by asset",
            "asset risk ranking",
            "riskiest assets",
            "most critical assets",
            "ativos com maior risco",
            "ativos com maior risco de anomalia",
            "quais ativos têm maior risco de anomalia",
            "quais ativos tem maior risco de anomalia",
            "maior risco de anomalia por ativo",
            "ranking de risco dos ativos",
            "ativos mais críticos",
            "ativos mais criticos",
            "equipamentos com maior risco",
            "equipamentos com maior risco de anomalia",
            "quais equipamentos têm maior risco",
            "quais equipamentos tem maior risco",
            "equipamentos mais críticos",
            "equipamentos mais criticos",
            "activos con mayor riesgo",
            "activos con mayor riesgo de anomalía",
            "activos con mayor riesgo de anomalia",
            "qué activos tienen mayor riesgo de anomalía",
            "que activos tienen mayor riesgo de anomalia",
            "mayor riesgo de anomalía por activo",
            "mayor riesgo de anomalia por activo",
            "ranking de riesgo de activos",
            "activos más críticos",
            "activos mas criticos",
            "equipos con mayor riesgo",
            "equipos con mayor riesgo de anomalía",
            "equipos con mayor riesgo de anomalia",
            "qué equipos tienen mayor riesgo",
            "que equipos tienen mayor riesgo",
            "equipos más críticos",
            "equipos mas criticos",
        ],
        "sql": """
            SELECT
                a.asset_id,
                a.asset_name,
                a.asset_type,
                a.location,
                COUNT(vm.measurement_id) AS total_measurements,
                ROUND(AVG(vm.rms_velocity), 3) AS avg_rms_velocity,
                ROUND(MAX(vm.peak_velocity), 3) AS max_peak_velocity,
                ROUND(AVG(sf.anomaly_score), 3) AS avg_anomaly_score,
                ROUND(MAX(sf.anomaly_score), 3) AS max_anomaly_score,
                ROUND(AVG(md.anomaly_probability), 3) AS avg_anomaly_probability,
                ROUND(MAX(md.anomaly_probability), 3) AS max_anomaly_probability,
                MAX(s.severity_level) AS max_recorded_severity
            FROM assets a
            JOIN vibration_measurements vm
                ON a.asset_id = vm.asset_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            LEFT JOIN spectral_features sf
                ON vm.measurement_id = sf.measurement_id
            LEFT JOIN ml_diagnostics md
                ON vm.measurement_id = md.measurement_id
            GROUP BY
                a.asset_id,
                a.asset_name,
                a.asset_type,
                a.location
            ORDER BY
                max_anomaly_probability DESC,
                max_anomaly_score DESC,
                avg_anomaly_probability DESC,
                avg_anomaly_score DESC
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