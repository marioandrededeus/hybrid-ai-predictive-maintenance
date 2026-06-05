# Hybrid Query Flow

This document describes the current query flow used in the **Ask the Database** section of the Hybrid AI for Predictive Maintenance project.

The goal of this architecture is to avoid using LLMs for every user question. Instead, the system first applies deterministic safety and routing layers before any future LLM-based fallback is considered.

## Purpose

The query flow is designed to:

- Keep the assistant focused on the predictive maintenance domain
- Prevent unsafe or unrelated prompts from reaching the SQL layer
- Reduce unnecessary LLM usage and token consumption
- Prefer known, safe SQL templates when possible
- Keep database access read-only and controlled
- Make the execution path transparent to the user

## Current Flow

```text
User Prompt
    ↓
Domain Guard
    ↓
Semantic Query Router
    ↓
Predefined SQL Template
    ↓
SQL Guard
    ↓
SQLite Database
    ↓
Streamlit Result
````

If no predefined SQL template is found, the system falls back to the controlled mock Text-to-SQL layer:

```text
User Prompt
    ↓
Domain Guard
    ↓
Semantic Query Router
    ↓
Mock Text-to-SQL
    ↓
SQL Guard
    ↓
SQLite Database
    ↓
Streamlit Result
```

## 1. Domain Guard

The Domain Guard validates whether the user prompt belongs to the expected project context.

It allows questions related to:

* Industrial predictive maintenance
* Vibration diagnostics
* Monitored assets
* Anomaly detection
* Risk analysis
* Maintenance recommendations
* Human validation

It blocks prompts outside the project context, such as general knowledge, weather, politics, travel, entertainment, finance, or unrelated uses of words like vibration.

Example blocked prompt:

```text
Vibration level of Etna vulcan
```

Even though the prompt contains the word `vibration`, it is not related to industrial predictive maintenance.

## 2. Semantic Query Router

The Semantic Query Router maps known user questions to predefined SQL templates.

This layer avoids unnecessary Text-to-SQL generation when the question can be answered by a safe and known query.

Examples of supported intents:

* Average anomaly score by scenario
* Average anomaly probability by scenario
* Highest risk scenarios
* Measurements requiring human validation
* Average RMS velocity by scenario
* Lubrication issues
* Structural looseness cases
* Monitored assets
* Anomaly risk by measurement

Example:

```text
Show average anomaly score by scenario
```

This prompt is routed to a predefined SQL template and does not require LLM usage.

## 3. Mock Text-to-SQL Fallback

When the prompt is inside the predictive maintenance domain but no predefined SQL template is available, the system falls back to the current mock Text-to-SQL layer.

This is still deterministic and controlled.

In the future, this fallback can be replaced by a real LLM-based Text-to-SQL component.

## 4. SQL Guard

The SQL Guard validates the generated SQL before execution.

It allows safe read-only `SELECT` queries and blocks unsafe operations such as:

* `CREATE`
* `ALTER`
* `DROP`
* `DELETE`
* `UPDATE`
* `INSERT`

This prevents destructive database operations.

## 5. SQLite Execution

Only validated SQL queries are executed against the SQLite database.

The current database contains pre-populated industrial maintenance data, including:

* Assets
* Scenarios
* Vibration measurements
* Spectral features
* ML diagnostics
* Human feedback

## 6. Streamlit Result

The Streamlit app displays:

* The execution path
* The matched intent, when available
* SQL validation status
* Generated SQL
* Query result

This makes the system transparent and easier to explain.

## Execution Path Transparency

The app shows whether the user question was:

* Blocked by the Domain Guard
* Answered by the Semantic Query Router
* Sent to the mock Text-to-SQL fallback
* Validated by the SQL Guard

Example:

```json
{
  "domain_guard": "passed",
  "semantic_router": "matched",
  "matched_intent": "average_anomaly_score_by_scenario",
  "text_to_sql_fallback": "not used",
  "llm_usage": "not used",
  "sql_guard": "passed"
}
```

## Why This Matters

This architecture supports a practical hybrid AI approach.

The system does not use LLMs as the first option. It uses deterministic layers whenever possible and reserves LLM-based reasoning for cases where it is truly needed.

This improves:

* Safety
* Cost control
* Explainability
* Reliability
* Maintainability

## Future Evolution

Future versions may include:

* Embedding-based semantic matching
* Vector database for known questions and SQL templates
* Real LLM-based Text-to-SQL fallback
* RAG-based technical explanation layer
* Agentic orchestration across database, ML, RAG, and human validation tools