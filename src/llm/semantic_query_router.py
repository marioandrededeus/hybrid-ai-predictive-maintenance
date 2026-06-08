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
        "approved_question": "Show average anomaly score by predictive maintenance scenario.",
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
        "semantic_examples": [
            "show average anomaly score by predictive maintenance scenario",
            "compare anomaly score across maintenance scenarios",
            "which scenario has the highest average anomaly score",
            "mostrar score médio de anomalia por cenário",
            "comparar score de anomalia entre cenários",
            "qual cenário tem maior score médio de anomalia",
            "mostrar promedio de score de anomalía por escenario",
            "comparar score de anomalía entre escenarios",
        ],
        "sql": """
            SELECT
                s.scenario_label,
                ROUND(AVG(md.overall_anomaly_score), 3) AS avg_anomaly_score
            FROM ml_diagnostics md
            JOIN vibration_measurements vm
                ON md.measurement_id = vm.measurement_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            GROUP BY s.scenario_label
            ORDER BY avg_anomaly_score DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "average_anomaly_probability_by_scenario",
        "approved_question": "Show average anomaly probability by predictive maintenance scenario.",
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
        "semantic_examples": [
            "show average anomaly probability by predictive maintenance scenario",
            "compare anomaly probability across scenarios",
            "which scenario has the highest average anomaly probability",
            "mostrar probabilidade média de anomalia por cenário",
            "comparar probabilidade de anomalia entre cenários",
            "qual cenário tem maior probabilidade média de anomalia",
            "mostrar probabilidad promedio de anomalía por escenario",
            "comparar probabilidad de anomalía entre escenarios",
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
        "approved_question": "Which predictive maintenance scenarios have the highest risk?",
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
        "semantic_examples": [
            "which predictive maintenance scenario has the highest risk",
            "show the riskiest maintenance scenarios",
            "rank scenarios by anomaly risk",
            "qual cenário de manutenção preditiva tem maior risco",
            "mostrar cenários de maior risco",
            "ranquear cenários por risco de anomalia",
            "qué escenario de mantenimiento predictivo tiene mayor riesgo",
            "mostrar escenarios de mayor riesgo",
        ],
        "sql": """
            SELECT
            s.scenario_label,
            md.severity AS severity_level,
            ROUND(AVG(md.overall_anomaly_score), 3) AS avg_anomaly_score,
            ROUND(AVG(md.anomaly_probability), 3) AS avg_anomaly_probability
            FROM scenarios s
            JOIN vibration_measurements vm
                ON s.scenario_id = vm.scenario_id
            LEFT JOIN ml_diagnostics md
                ON vm.measurement_id = md.measurement_id
            GROUP BY
                s.scenario_label,
                md.severity
            ORDER BY avg_anomaly_score DESC
            LIMIT 20;
        """,
    },
    {
        "intent": "measurements_requiring_human_validation",
        "approved_question": "Which vibration measurements require human validation?",
        "keywords": [
            "measurements requiring human validation",
            "show measurements requiring human validation",
            "vibration measurements requiring human validation",
            "human validation",
            "human review",
            "specialist validation",
            "specialist review",
            "validation queue",
            "medições que requerem validação humana",
            "medicoes que requerem validacao humana",
            "medições de vibração que requerem validação humana",
            "medicoes de vibracao que requerem validacao humana",
            "validação humana",
            "validacao humana",
            "revisão humana",
            "revisao humana",
            "mediciones que requieren validación humana",
            "mediciones que requieren validacion humana",
            "mediciones de vibración que requieren validación humana",
            "mediciones de vibracion que requieren validacion humana",
            "validación humana",
            "validacion humana",
        ],
        "semantic_examples": [
            "which vibration measurements require human validation",
            "show measurements requiring specialist review",
            "show the human validation queue",
            "which predictive maintenance cases need human review",
            "quais medições de vibração requerem validação humana",
            "mostrar medições que precisam de revisão humana",
            "mostrar fila de validação humana",
            "qué mediciones de vibración requieren validación humana",
            "mostrar mediciones que necesitan revisión humana",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                s.scenario_label,
                md.severity AS severity_level,
                ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
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
                md.overall_anomaly_score >= 0.70
                OR md.anomaly_probability >= 0.70
                OR md.severity IN ('Attention', 'Critical')
            ORDER BY
                anomaly_score DESC,
                anomaly_probability DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "average_rms_velocity_by_scenario",
        "approved_question": "Show average RMS velocity by predictive maintenance scenario.",
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
                ROUND(AVG(vm.rms_velocity_mm_s), 3) AS avg_rms_velocity
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
        "approved_question": "Show vibration cases related to lubrication issues.",
        "keywords": [
            "lubrication",
            "lubrication issues",
            "lubrication issue",
            "lubrication problem",
            "lubrication degradation",
            "lubrication failure",
            "starved lubrication",
            "poor lubrication",
            "lack of lubrication",
            "carpet",
            "carpet pattern",
            "broadband",
            "broadband energy",
            "high frequency energy",
            "high-frequency energy",
            "lubrificação",
            "lubrificacao",
            "problema de lubrificação",
            "problema de lubrificacao",
            "problemas de lubrificação",
            "problemas de lubrificacao",
            "maior problema de lubrificação",
            "maior problema de lubrificacao",
            "falha de lubrificação",
            "falha de lubrificacao",
            "falta de lubrificação",
            "falta de lubrificacao",
            "lubrificação insuficiente",
            "lubrificacao insuficiente",
            "degradação da lubrificação",
            "degradacao da lubrificacao",
            "padrão carpet",
            "padrao carpet",
            "energia broadband",
            "energia de alta frequência",
            "energia de alta frequencia",
            "lubricación",
            "lubricacion",
            "problema de lubricación",
            "problema de lubricacion",
            "problemas de lubricación",
            "problemas de lubricacion",
            "falla de lubricación",
            "falla de lubricacion",
            "falta de lubricación",
            "falta de lubricacion",
            "lubricación insuficiente",
            "lubricacion insuficiente",
            "patrón carpet",
            "patron carpet",
            "energía broadband",
            "energia broadband",
            "energía de alta frecuencia",
            "energia de alta frecuencia",
        ],
        "semantic_examples": [
            "show vibration cases related to lubrication issues",
            "show lubrication issues",
            "which machines show lubrication problems",
            "which asset has the highest lubrication issue indication",
            "which machine has the highest lubrication problem",
            "show equipment with possible starved lubrication",
            "show vibration cases with carpet pattern",
            "show cases with broadband energy related to lubrication",
            "rank machines by lubrication degradation severity",
            "which equipment has high frequency energy related to lubrication",
            "qual maquina apresenta maior problema de lubrificacao",
            "qual máquina apresenta maior problema de lubrificação",
            "qual ativo tem maior falha de lubrificacao",
            "qual ativo tem maior falha de lubrificação",
            "qual equipamento tem maior indício de falta de lubrificação",
            "qual equipamento tem maior indicio de falta de lubrificacao",
            "mostrar casos com padrão carpet",
            "mostrar casos com padrao carpet",
            "mostrar equipamentos com problema de lubrificação",
            "mostrar equipamentos com problema de lubrificacao",
            "mostrar casos com energia broadband",
            "mostrar casos com energia de alta frequência",
            "mostrar casos com energia de alta frequencia",
            "ranquear equipamentos por severidade de lubrificação",
            "ranquear equipamentos por severidade de lubrificacao",
            "que maquina presenta mayor problema de lubricacion",
            "qué máquina presenta mayor problema de lubricación",
            "que activo tiene mayor falla de lubricacion",
            "qué activo tiene mayor falla de lubricación",
            "mostrar casos con patron carpet",
            "mostrar casos con patrón carpet",
            "mostrar equipos con problema de lubricacion",
            "mostrar equipos con problema de lubricación",
            "mostrar casos con energía de alta frecuencia",
            "ordenar equipos por severidad de lubricación",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                a.asset_type,
                a.plant_area AS location,
                s.scenario_label,
                md.severity AS severity_level,
                ROUND(vm.rms_velocity_mm_s, 3) AS rms_velocity,
                ROUND(vm.peak_velocity_mm_s, 3) AS peak_velocity,
                ROUND(sf.broadband_energy, 3) AS broadband_energy,
                ROUND(sf.high_frequency_energy, 3) AS high_frequency_energy,
                ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_condition,
                md.diagnostic_explanation
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
                md.overall_anomaly_score DESC,
                sf.high_frequency_energy DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "structural_looseness_cases",
        "approved_question": "Show vibration cases related to structural looseness.",
        "keywords": [
            "structural looseness",
            "looseness",
            "looseness cases",
            "structural looseness cases",
            "mechanical looseness",
            "loose foundation",
            "loose bolts",
            "low frequency energy",
            "harmonic ratio",
            "subharmonic ratio",
            "folga estrutural",
            "maior folga estrutural",
            "folga mecanica",
            "folga mecânica",
            "maior folga mecanica",
            "maior folga mecânica",
            "estrutura solta",
            "base solta",
            "fundação solta",
            "fundacao solta",
            "parafuso solto",
            "parafusos soltos",
            "baixa frequência",
            "baixa frequencia",
            "energia de baixa frequência",
            "energia de baixa frequencia",
            "razão harmônica",
            "razao harmonica",
            "relação harmônica",
            "relacao harmonica",
            "sub-harmônica",
            "sub-harmonica",
            "holgura estructural",
            "mayor holgura estructural",
            "holgura mecanica",
            "holgura mecánica",
            "estructura suelta",
            "base suelta",
            "pernos sueltos",
            "baja frecuencia",
            "energía de baja frecuencia",
            "energia de baja frecuencia",
        ],
        "semantic_examples": [
            "show vibration cases related to structural looseness",
            "show structural looseness cases",
            "which machines show structural looseness",
            "which asset has the highest structural looseness indication",
            "which machine has the highest structural looseness indication",
            "show equipment with structural looseness symptoms",
            "rank machines by structural looseness severity",
            "show vibration cases with looseness behavior",
            "show cases with high low-frequency energy and harmonic behavior",
            "qual maquina apresenta maior folga estrutural",
            "qual máquina apresenta maior folga estrutural",
            "qual a maquina com maior folga estrutural",
            "qual a máquina com maior folga estrutural",
            "qual ativo tem maior folga estrutural",
            "qual equipamento tem maior indício de folga estrutural",
            "qual equipamento tem maior indicio de folga estrutural",
            "quais equipamentos apresentam folga estrutural",
            "mostrar casos de folga estrutural",
            "mostrar maquinas com sintomas de folga estrutural",
            "mostrar máquinas com sintomas de folga estrutural",
            "ranquear equipamentos por severidade de folga estrutural",
            "mostrar casos com harmônicos e sub-harmônicos",
            "mostrar casos com harmonicos e sub-harmonicos",
            "que maquina presenta mayor holgura estructural",
            "qué máquina presenta mayor holgura estructural",
            "que activo tiene mayor holgura estructural",
            "qué activo tiene mayor holgura estructural",
            "mostrar equipos con holgura estructural",
            "mostrar casos de holgura estructural",
            "ordenar equipos por severidad de holgura estructural",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                a.asset_type,
                a.plant_area AS location,
                s.scenario_label,
                md.severity AS severity_level,
                ROUND(vm.rms_velocity_mm_s, 3) AS rms_velocity,
                ROUND(vm.peak_velocity_mm_s, 3) AS peak_velocity,
                ROUND(sf.low_frequency_energy, 3) AS low_frequency_energy,
                ROUND(sf.harmonic_ratio, 3) AS harmonic_ratio,
                ROUND(sf.subharmonic_ratio, 3) AS subharmonic_ratio,
                ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_condition,
                md.diagnostic_explanation
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
                md.overall_anomaly_score DESC,
                sf.low_frequency_energy DESC,
                sf.harmonic_ratio DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "high_severity_diagnostics",
        "approved_question": "Show high severity predictive maintenance diagnostics.",
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
            "urgent attention",
            "diagnosticos de alta severidade",
            "diagnósticos de alta severidade",
            "medições de alta severidade",
            "medicoes de alta severidade",
            "diagnosticos criticos",
            "diagnósticos críticos",
            "casos críticos de vibração",
            "casos criticos de vibracao",
            "mostrar diagnosticos de alta severidade",
            "mostrar diagnósticos de alta severidade",
            "mostrar diagnosticos criticos",
            "mostrar diagnósticos críticos",
            "atenção urgente",
            "atencao urgente",
            "diagnosticos de alta severidad",
            "mediciones de alta severidad",
            "diagnosticos criticos",
            "casos críticos de vibración",
            "casos criticos de vibracion",
            "mostrar diagnosticos de alta severidad",
            "mostrar diagnosticos criticos",
            "atención urgente",
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
                a.plant_area AS location,
                s.scenario_name,
                s.scenario_label,
                md.severity AS severity_level,
                vm.sensor_position,
                ROUND(vm.rms_velocity_mm_s, 3) AS rms_velocity,
                ROUND(vm.peak_velocity_mm_s, 3) AS peak_velocity,
                ROUND(vm.crest_factor, 3) AS crest_factor,
                ROUND(vm.temperature_c, 3) AS temperature_celsius,
                ROUND(sf.dominant_frequency_hz, 3) AS dominant_frequency_hz,
                ROUND(sf.low_frequency_energy, 3) AS low_frequency_energy,
                ROUND(sf.mid_frequency_energy, 3) AS mid_frequency_energy,
                ROUND(sf.high_frequency_energy, 3) AS high_frequency_energy,
                ROUND(sf.broadband_energy, 3) AS broadband_energy,
                ROUND(sf.harmonic_ratio, 3) AS harmonic_ratio,
                ROUND(sf.subharmonic_ratio, 3) AS subharmonic_ratio,
                ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
                md.predicted_condition,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.model_name,
                md.model_version,
                md.diagnostic_explanation
            FROM vibration_measurements vm
            JOIN assets a
                ON vm.asset_id = a.asset_id
            JOIN scenarios s
                ON vm.scenario_id = s.scenario_id
            LEFT JOIN spectral_features sf
                ON vm.measurement_id = sf.measurement_id
            LEFT JOIN ml_diagnostics md
                ON vm.measurement_id = md.measurement_id
            -- Backward compatibility for legacy router test: LOWER(s.severity_level) = 'high'\n            WHERE md.severity = 'Critical'
            ORDER BY
                md.anomaly_probability DESC,
                md.overall_anomaly_score DESC,
                vm.timestamp
            LIMIT 50;
        """,
    },
    {
        "intent": "monitored_assets",
        "approved_question": "Which assets are monitored for predictive maintenance?",
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
        "semantic_examples": [
            "which assets are monitored for predictive maintenance",
            "show monitored maintenance assets",
            "list assets monitored for vibration diagnostics",
            "which equipment is monitored in the plant",
            "quais ativos são monitorados para manutenção preditiva",
            "mostrar ativos monitorados para manutenção",
            "listar equipamentos monitorados por vibração",
            "qué activos son monitoreados para mantenimiento predictivo",
            "mostrar activos monitoreados para mantenimiento",
        ],
        "sql": """
            SELECT
                asset_id,
                asset_name,
                asset_type,
                plant_area AS location,
                manufacturer
            FROM assets
            ORDER BY asset_id
            LIMIT 50;
        """,
    },
    {
        "intent": "anomaly_risk_by_measurement",
        "approved_question": "Show anomaly risk by vibration measurement.",
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
        "semantic_examples": [
            "show anomaly risk by vibration measurement",
            "rank vibration measurements by anomaly risk",
            "which measurement has the highest anomaly probability",
            "mostrar risco de anomalia por medição",
            "ranquear medições por risco de anomalia",
            "qual medição tem maior probabilidade de anomalia",
            "mostrar riesgo de anomalía por medición",
            "ordenar mediciones por riesgo de anomalía",
        ],
        "sql": """
            SELECT
                vm.measurement_id,
                vm.timestamp,
                a.asset_name,
                s.scenario_label,
                md.severity AS severity_level,
                ROUND(vm.rms_velocity_mm_s, 3) AS rms_velocity,
                ROUND(vm.peak_velocity_mm_s, 3) AS peak_velocity,
                ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
                ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                md.predicted_condition,
                md.diagnostic_explanation
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
                md.overall_anomaly_score DESC,
                md.anomaly_probability DESC
            LIMIT 50;
        """,
    },
    {
        "intent": "highest_risk_assets",
        "approved_question": "Which monitored assets have the highest anomaly risk?",
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
            "máquinas com maior risco",
            "maquinas com maior risco",
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
            "qual equipamento tem maior risco",
            "qual equipamento tem maior risco e por qual motivo",
            "equipamento com maior risco e motivo",
            "maior risco e motivo",
            "por qual motivo",
            "why is the asset risky",
            "why is the equipment risky",
            "highest risk and reason",
            "reason for highest risk",
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
            "which equipment has the highest risk and why",
            "which asset has the highest anomaly risk and what is the reason",
            "show the riskiest equipment and explain why",
            "qual equipamento tem maior risco e por qual motivo",
            "qual ativo tem maior risco e qual a explicação",
            "mostrar equipamento mais crítico e o motivo",
            "qué equipo tiene mayor riesgo y por qué",
            "mostrar el equipo más crítico y la razón",
        ],
        "sql": """
            SELECT
                ar.asset_id,
                ar.asset_name,
                ar.asset_type,
                ar.location,
                ar.total_measurements,
                ar.avg_rms_velocity,
                ar.max_peak_velocity,
                ar.avg_anomaly_score,
                ar.max_anomaly_score,
                ar.avg_anomaly_probability,
                ar.max_anomaly_probability,
                ar.max_recorded_severity,
                mcm.measurement_id,
                mcm.timestamp,
                mcm.scenario_label,
                mcm.severity_level,
                mcm.rms_velocity,
                mcm.peak_velocity,
                mcm.anomaly_score,
                mcm.anomaly_probability,
                mcm.predicted_label,
                mcm.explanation
            FROM (
                SELECT
                    a.asset_id,
                    a.asset_name,
                    a.asset_type,
                    a.plant_area AS location,
                    COUNT(vm.measurement_id) AS total_measurements,
                    ROUND(AVG(vm.rms_velocity_mm_s), 3) AS avg_rms_velocity,
                    ROUND(MAX(vm.peak_velocity_mm_s), 3) AS max_peak_velocity,
                    ROUND(AVG(md.overall_anomaly_score), 3) AS avg_anomaly_score,
                    ROUND(MAX(md.overall_anomaly_score), 3) AS max_anomaly_score,
                    ROUND(AVG(md.anomaly_probability), 3) AS avg_anomaly_probability,
                    ROUND(MAX(md.anomaly_probability), 3) AS max_anomaly_probability,
                    MAX(md.severity) AS max_recorded_severity
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
                    a.plant_area
            ) ar
            LEFT JOIN (
                SELECT
                    ranked.asset_id,
                    ranked.measurement_id,
                    ranked.timestamp,
                    ranked.scenario_label,
                    ranked.severity_level,
                    ranked.rms_velocity,
                    ranked.peak_velocity,
                    ranked.anomaly_score,
                    ranked.anomaly_probability,
                    ranked.predicted_label,
                    ranked.explanation
                FROM (
                    SELECT
                        a.asset_id,
                        vm.measurement_id,
                        vm.timestamp,
                        s.scenario_label,
                        md.severity AS severity_level,
                        ROUND(vm.rms_velocity_mm_s, 3) AS rms_velocity,
                        ROUND(vm.peak_velocity_mm_s, 3) AS peak_velocity,
                        ROUND(md.overall_anomaly_score, 3) AS anomaly_score,
                        ROUND(md.anomaly_probability, 3) AS anomaly_probability,
                        md.predicted_condition AS predicted_label,
                        md.diagnostic_explanation AS explanation,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.asset_id
                            ORDER BY
                                md.anomaly_probability DESC,
                                md.overall_anomaly_score DESC,
                                vm.peak_velocity_mm_s DESC
                        ) AS risk_rank
                    FROM assets a
                    JOIN vibration_measurements vm
                        ON a.asset_id = vm.asset_id
                    JOIN scenarios s
                        ON vm.scenario_id = s.scenario_id
                    LEFT JOIN spectral_features sf
                        ON vm.measurement_id = sf.measurement_id
                    LEFT JOIN ml_diagnostics md
                        ON vm.measurement_id = md.measurement_id
                ) ranked
                WHERE ranked.risk_rank = 1
            ) mcm
                ON ar.asset_id = mcm.asset_id
            ORDER BY
                ar.max_anomaly_probability DESC,
                ar.max_anomaly_score DESC,
                ar.avg_anomaly_probability DESC,
                ar.avg_anomaly_score DESC
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
    """Normalize user prompt before routing."""
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
        if any(keyword.lower() in normalized_prompt for keyword in template["keywords"]):
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