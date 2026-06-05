# Supported Multilingual Questions

This document lists representative questions supported by the Semantic Query Router in the Hybrid AI for Predictive Maintenance project.

The router maps user questions in English, Portuguese, and Spanish to predefined safe SQL templates.

The goal is not to provide open-ended natural language understanding yet, but to support controlled multilingual access to industrial maintenance analytics.

## How It Works

The flow follows a deterministic safety-first architecture:

1. The Domain Guard checks whether the question belongs to the predictive maintenance domain.
2. The Semantic Query Router tries to match the question to a predefined intent.
3. If a template is matched, a safe SQL query is selected.
4. The SQL Guard validates that only safe SELECT queries are executed.
5. The database returns structured results.
6. No LLM is required for the deterministic route.

## Supported Languages

The current semantic router supports representative questions in:

- English
- Portuguese, Brazil
- Spanish

## Supported Question Groups

## 1. Average anomaly score by scenario

Intent:

```text
average_anomaly_score_by_scenario
````

Examples:

```text
average anomaly score by scenario
media do score de anomalia por cenario
promedio del score de anomalia por escenario
```

## 2. Average anomaly probability by scenario

Intent:

```text
average_anomaly_probability_by_scenario
```

Examples:

```text
average anomaly probability by scenario
probabilidade de anomalia por cenario
probabilidad de anomalia por escenario
```

## 3. Highest risk scenarios

Intent:

```text
highest_risk_scenarios
```

Examples:

```text
which scenario has the highest risk
qual cenario tem maior risco
que escenario tiene mayor riesgo
```

## 4. Measurements requiring human validation

Intent:

```text
measurements_requiring_human_validation
```

Examples:

```text
measurements requiring human validation
medicoes que requerem validacao humana
mediciones que requieren validacion humana
```

## 5. Average RMS velocity by scenario

Intent:

```text
average_rms_velocity_by_scenario
```

Examples:

```text
average rms velocity by scenario
media da velocidade rms por cenario
promedio de velocidad rms por escenario
```

## 6. Lubrication issues

Intent:

```text
lubrication_issues
```

Examples:

```text
lubrication issues
problemas de lubrificacao
problemas de lubricacion
```

## 7. Structural looseness cases

Intent:

```text
structural_looseness_cases
```

Examples:

```text
structural looseness cases
casos de folga estrutural
casos de holgura estructural
```

## 8. High severity diagnostics

Intent:

```text
high_severity_diagnostics
```

Examples:

```text
show high severity diagnostics
mostrar diagnosticos de alta severidade
mostrar diagnosticos de alta severidad
```

## 9. Monitored assets

Intent:

```text
monitored_assets
```

Examples:

```text
which assets are monitored
quais ativos sao monitorados
que activos son monitoreados
```

## 10. Anomaly risk by measurement

Intent:

```text
anomaly_risk_by_measurement
```

Examples:

```text
anomaly risk by measurement
risco de anomalia por medicao
riesgo de anomalia por medicion
```

## 11. Highest risk assets

Intent:

```text
highest_risk_assets
```

Examples:

```text
which assets have the highest anomaly risk
quais ativos tem maior risco de anomalia
que activos tienen mayor riesgo de anomalia
```

## Design Decision

This multilingual support is intentionally deterministic.

Instead of allowing an LLM to freely generate SQL, the system uses multilingual keywords to route known business questions to safe SQL templates.

This approach is useful for industrial environments because it improves:

* auditability
* safety
* repeatability
* explainability
* multilingual accessibility

## Future Improvements

Possible next steps:

* add accent-insensitive text normalization
* add more Portuguese and Spanish variations
* add a supported questions panel in the Streamlit app
* connect these documented intents to the future RAG layer
* allow the LLM to explain results while keeping SQL generation controlled