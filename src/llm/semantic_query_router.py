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
            "anomaly score by scenario",
            "show average anomaly score by scenario",
            "score de anomalia por cenário",
            "score de anomalia por cenario",
            "média de anomalia por cenário",
            "media de anomalia por cenario",
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
            "anomaly probability by scenario",
            "show average anomaly probability by scenario",
            "probabilidade de anomalia por cenário",
            "probabilidade de anomalia por cenario",
            "média de probabilidade de anomalia por cenário",
            "media de probabilidade de anomalia por cenario",
            "probabilidad de anomalía por escenario",
            "probabilidad de anomalia por escenario",
            "promedio de probabilidad de anomalía por escenario",
            "promedio de probabilidad de anomalia por escenario",
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
            "riskiest scenario",
            "which scenario has the highest risk",
            "highest risk scenarios",
            "show highest risk scenarios",
            "cenário de maior risco",
            "cenario de maior risco",
            "cenários de maior risco",
            "cenarios de maior risco",
            "qual cenário tem maior risco",
            "qual cenario tem maior risco",
            "mostrar cenários de maior risco",
            "mostrar cenarios de maior risco",
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
            "measurements requiring human validation",
            "show measurements requiring human validation",
            "vibration measurements requiring human validation",
            "medições que requerem validação humana",
            "medicoes que requerem validacao humana",
            "medições de vibração que requerem validação humana",
            "medicoes de vibracao que requerem validacao humana",
            "mediciones que requieren validación humana",
            "mediciones que requieren validacion humana",
            "mediciones de vibración que requieren validación humana",
            "mediciones de vibracion que requieren validacion humana",
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
            "average rms velocity by predictive maintenance scenario",
            "rms velocity by predictive maintenance scenario",
            "average vibration rms velocity by maintenance scenario",
            "show average rms velocity by maintenance scenario",
            "velocidade rms por cenário de manutenção preditiva",
            "velocidade rms por cenario de manutencao preditiva",
            "velocidade rms de vibração por cenário de manutenção",
            "velocidade rms de vibracao por cenario de manutencao",
            "média da velocidade rms por cenário de manutenção",
            "media da velocidade rms por cenario de manutencao",
            "velocidad rms por escenario de mantenimiento predictivo",
            "velocidad rms de vibración por escenario de mantenimiento",
            "velocidad rms de vibracion por escenario de mantenimiento",
            "promedio de velocidad rms por escenario de mantenimiento",
        ],
                        "semantic_examples": [
            "average rms velocity by predictive maintenance scenario",
            "compare vibration intensity across scenarios",
            "which scenario has the highest average vibration level",
            "show mean rms velocity for each diagnostic scenario",
            "compare rms vibration between failure patterns",
            "show vibration severity grouped by scenario",
            "comparar intensidade de vibração entre cenários",
            "qual cenário tem maior nível médio de vibração",
            "mostrar velocidade rms média por cenário diagnóstico",
            "comparar vibração rms entre padrões de falha",
            "mostrar severidade de vibração agrupada por cenário",
            "comparar intensidad de vibración entre escenarios",
            "qué escenario tiene mayor nivel promedio de vibración",
            "mostrar velocidad rms promedio por escenario diagnóstico",
            "comparar vibración rms entre patrones de falla",
            "mostrar severidad de vibración agrupada por escenario",
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
            "lubrication failure",
            "high frequency energy in lubrication issue",
            "broadband energy in lubrication issue",
            "problemas de lubrificação em vibração",
            "problemas de lubrificacao em vibracao",
            "problema de lubrificação em vibração",
            "problema de lubrificacao em vibracao",
            "degradação da lubrificação em vibração",
            "degradacao da lubrificacao em vibracao",
            "falha de lubrificação em vibração",
            "falha de lubrificacao em vibracao",
            "padrão carpet",
            "padrao carpet",
            "energia de alta frequência em lubrificação",
            "energia de alta frequencia em lubrificacao",
            "energia broadband em lubrificação",
            "energia broadband em lubrificacao",
            "problemas de lubricación en vibración",
            "problemas de lubricacion en vibracion",
            "problema de lubricación en vibración",
            "problema de lubricacion en vibracion",
            "degradación de lubricación en vibración",
            "degradacion de lubricacion en vibracion",
            "falla de lubricación en vibración",
            "falla de lubricacion en vibracion",
            "patrón carpet",
            "patron carpet",
            "energía de alta frecuencia en lubricación",
            "energia de alta frecuencia en lubricacion",
            "energía broadband en lubricación",
            "energia broadband en lubricacion",
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
            "structural looseness cases",
            "low frequency energy in structural looseness",
            "harmonic ratio in structural looseness",
            "subharmonic ratio in structural looseness",
            "folga estrutural em vibração",
            "folga estrutural em vibracao",
            "casos de folga estrutural em vibração",
            "casos de folga estrutural em vibracao",
            "energia de baixa frequência em folga estrutural",
            "energia de baixa frequencia em folga estrutural",
            "razao harmonica em folga estrutural",
            "relacao harmonica em folga estrutural",
            "razao sub-harmonica em folga estrutural",
            "relacao sub-harmonica em folga estrutural",
            "holgura estructural en vibración",
            "holgura estructural en vibracion",
            "casos de holgura estructural en vibración",
            "casos de holgura estructural en vibracion",
            "energía de baja frecuencia en holgura estructural",
            "energia de baja frecuencia en holgura estructural",
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
            "high severity vibration measurements",
            "critical diagnostic cases",
            "critical diagnostics",
            "critical maintenance cases",
            "critical vibration cases",
            "which cases are high severity",
            "show high severity diagnostics",
            "show critical diagnostics",
            "diagnosticos de alta severidade",
            "medições de alta severidade",
            "medicoes de alta severidade",
            "diagnosticos criticos",
            "casos críticos de vibração",
            "casos criticos de vibracao",
            "mostrar diagnosticos de alta severidade",
            "mostrar diagnosticos criticos",
            "diagnosticos de alta severidad",
            "mediciones de alta severidad",
            "diagnosticos criticos",
            "casos críticos de vibración",
            "casos criticos de vibracion",
            "mostrar diagnosticos de alta severidad",
            "mostrar diagnosticos criticos",
        ],
                "semantic_examples": [
            "show high severity diagnostics",
            "show the most critical diagnostic cases",
            "which diagnostics require urgent attention",
            "list severe predictive maintenance cases",
            "show measurements classified as high severity",
            "which cases should be prioritized by maintenance",
            "mostrar diagnósticos de alta severidade",
            "mostrar os casos diagnósticos mais críticos",
            "quais diagnósticos exigem atenção urgente",
            "listar casos severos de manutenção preditiva",
            "mostrar medições classificadas como alta severidade",
            "mostrar diagnósticos de alta severidad",
            "mostrar los casos diagnósticos más críticos",
            "qué diagnósticos requieren atención urgente",
            "listar casos severos de mantenimiento predictivo",
            "mostrar mediciones clasificadas como alta severidad",
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
            "which assets are monitored for predictive maintenance",
            "monitored assets in predictive maintenance",
            "show monitored maintenance assets",
            "list monitored maintenance assets",
            "industrial assets monitored for vibration diagnostics",
            "which equipment is monitored for predictive maintenance",
            "monitored equipment in vibration diagnostics",
            "quais ativos são monitorados para manutenção preditiva",
            "quais ativos sao monitorados para manutencao preditiva",
            "ativos monitorados para manutenção preditiva",
            "ativos monitorados para manutencao preditiva",
            "mostrar ativos monitorados para manutenção",
            "mostrar ativos monitorados para manutencao",
            "listar ativos monitorados para manutenção",
            "listar ativos monitorados para manutencao",
            "equipamentos monitorados para manutenção preditiva",
            "equipamentos monitorados para manutencao preditiva",
            "qué activos son monitoreados para mantenimiento predictivo",
            "que activos son monitoreados para mantenimiento predictivo",
            "activos monitoreados para mantenimiento predictivo",
            "mostrar activos monitoreados para mantenimiento",
            "listar activos monitoreados para mantenimiento",
            "equipos monitoreados para mantenimiento predictivo",
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
            "measurement risk",
            "risk per measurement",
            "show anomaly risk by measurement",
            "risco de anomalia por medição",
            "risco de anomalia por medicao",
            "risco por medição",
            "risco por medicao",
            "anomalia por medição",
            "anomalia por medicao",
            "mostrar risco de anomalia por medição",
            "mostrar risco de anomalia por medicao",
            "risco da medição",
            "risco da medicao",
            "riesgo de anomalía por medición",
            "riesgo de anomalia por medicion",
            "riesgo por medición",
            "riesgo por medicion",
            "anomalía por medición",
            "anomalia por medicion",
            "mostrar riesgo de anomalía por medición",
            "mostrar riesgo de anomalia por medicion",
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
            "most critical maintenance assets",
            "ativos com maior risco de anomalia",
            "quais ativos têm maior risco de anomalia",
            "quais ativos tem maior risco de anomalia",
            "maior risco de anomalia por ativo",
            "ranking de risco dos ativos",
            "ativos mais críticos",
            "ativos mais criticos",
            "equipamentos com maior risco de anomalia",
            "quais equipamentos têm maior risco de anomalia",
            "quais equipamentos tem maior risco de anomalia",
            "equipamentos mais críticos",
            "equipamentos mais criticos",
            "activos con mayor riesgo de anomalía",
            "activos con mayor riesgo de anomalia",
            "qué activos tienen mayor riesgo de anomalía",
            "que activos tienen mayor riesgo de anomalia",
            "ranking de riesgo de activos",
            "activos más críticos",
            "activos mas criticos",
            "equipos con mayor riesgo de anomalía",
            "equipos con mayor riesgo de anomalia",
            "qué equipos tienen mayor riesgo de anomalía",
            "que equipos tienen mayor riesgo de anomalia",
            "equipos más críticos",
            "equipos mas criticos",
        ],
                "semantic_examples": [
            "which assets have the highest anomaly risk",
            "which machines look most critical",
            "show equipment with abnormal behavior",
            "rank assets by operational risk",
            "which monitored assets require attention first",
            "show the riskiest equipment in the plant",
            "quais ativos parecem mais críticos",
            "quais máquinas estão com maior risco",
            "mostrar equipamentos com comportamento anormal",
            "ranquear ativos por risco operacional",
            "quais ativos precisam de atenção primeiro",
            "que equipos parecen más críticos",
            "que máquinas tienen mayor riesgo",
            "mostrar equipos con comportamiento anormal",
            "ordenar activos por riesgo operativo",
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

SUPPORTED_DEMO_QUESTIONS = [
    {
        "category": "Risk ranking",
        "english": "which assets have the highest anomaly risk",
        "portuguese": "quais ativos tem maior risco de anomalia",
        "spanish": "que activos tienen mayor riesgo de anomalia",
    },
    {
        "category": "Risk ranking",
        "english": "which scenario has the highest risk",
        "portuguese": "qual cenario tem maior risco",
        "spanish": "que escenario tiene mayor riesgo",
    },
    {
        "category": "Diagnostics",
        "english": "show high severity diagnostics",
        "portuguese": "mostrar diagnosticos de alta severidade",
        "spanish": "mostrar diagnosticos de alta severidad",
    },
    {
        "category": "Anomaly metrics",
        "english": "anomaly score by scenario",
        "portuguese": "score de anomalia por cenario",
        "spanish": "score de anomalia por escenario",
    },
    {
        "category": "Anomaly metrics",
        "english": "anomaly probability by scenario",
        "portuguese": "probabilidade de anomalia por cenario",
        "spanish": "probabilidad de anomalia por escenario",
    },
    {
        "category": "Vibration metrics",
        "english": "average rms velocity by predictive maintenance scenario",
        "portuguese": "velocidade rms por cenario de manutencao preditiva",
        "spanish": "velocidad rms por escenario de mantenimiento predictivo",
    },
    {
        "category": "Failure patterns",
        "english": "lubrication issues",
        "portuguese": "problemas de lubrificacao em vibracao",
        "spanish": "problemas de lubricacion en vibracion",
    },
    {
        "category": "Failure patterns",
        "english": "structural looseness cases",
        "portuguese": "casos de folga estrutural em vibracao",
        "spanish": "casos de holgura estructural en vibracion",
    },
    {
        "category": "Human validation",
        "english": "measurements requiring human validation",
        "portuguese": "medicoes que requerem validacao humana",
        "spanish": "mediciones que requieren validacion humana",
    },
    {
        "category": "Assets",
        "english": "which assets are monitored for predictive maintenance",
        "portuguese": "quais ativos sao monitorados para manutencao preditiva",
        "spanish": "que activos son monitoreados para mantenimiento predictivo",
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

    Routing order:
    1. Domain Guard
    2. Keyword-based semantic router
    3. Embedding-based semantic router fallback
    """

    domain_validation = get_domain_guard_response(prompt)

    if not domain_validation["is_allowed"]:
        return {
            "status": "blocked",
            "intent": None,
            "sql": None,
            "message": domain_validation["message"],
            "routing_method": "domain_guard",
            "similarity_score": None,
            "matched_example": None,
        }

    normalized_prompt = normalize_prompt(prompt)

    for template in QUERY_TEMPLATES:
        if any(keyword in normalized_prompt for keyword in template["keywords"]):
            return {
                "status": "matched",
                "intent": template["intent"],
                "sql": template["sql"],
                "message": "Prompt matched a predefined SQL template.",
                "routing_method": "keyword_match",
                "similarity_score": None,
                "matched_example": None,
            }

    from src.llm import embedding_router

    embedding_response = embedding_router.route_prompt_by_embedding(prompt)

    if embedding_response["status"] == "matched":
        return {
            "status": "matched",
            "intent": embedding_response["intent"],
            "sql": embedding_response["sql"],
            "message": "Prompt matched an approved SQL template using embedding similarity.",
            "routing_method": "embedding_similarity",
            "similarity_score": embedding_response["similarity_score"],
            "matched_example": embedding_response["matched_example"],
        }

    return {
        "status": "not_matched",
        "intent": None,
        "sql": None,
        "message": (
            "Prompt is inside the predictive maintenance domain, "
            "but no approved SQL template was found."
        ),
        "routing_method": "not_matched",
        "similarity_score": embedding_response.get("similarity_score"),
        "matched_example": embedding_response.get("matched_example"),
    }

def get_supported_query_examples() -> list[str]:
    """Return curated example questions for the Streamlit interface."""
    examples = []

    for question_group in SUPPORTED_DEMO_QUESTIONS:
        examples.extend(
            [
                question_group["english"],
                question_group["portuguese"],
                question_group["spanish"],
            ]
        )

    return examples

def get_supported_demo_questions() -> list[dict]:
    """Return curated multilingual questions grouped for UI display."""
    return SUPPORTED_DEMO_QUESTIONS