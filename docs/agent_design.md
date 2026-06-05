Perfeito. Vamos fechar este chat com a documentação do agente.

# Agent Design

This document describes the current deterministic maintenance agent used in the **Hybrid AI for Predictive Maintenance** project.

The goal of this agent is to provide a simple and explainable orchestration layer before introducing real LLM-based agentic workflows.

## Purpose

The maintenance agent coordinates existing project tools to generate a structured diagnostic response for a selected industrial scenario.

In the current version, the agent does not use a real LLM.

It is deterministic, transparent and easy to test.

## Current Agent Flow

```text
Scenario ID
    ↓
Maintenance Agent
    ↓
Scenario Retrieval
    ↓
Measurement Retrieval
    ↓
Scenario Summary
    ↓
Rule-Based Recommendations
    ↓
Agent Reasoning
````

## Current Implementation

Main file:

```text
src/agents/maintenance_agent.py
```

Main function:

```python
run_maintenance_agent(scenario_id: int) -> dict
```

The agent currently performs the following steps:

1. Receives a `scenario_id`
2. Retrieves the corresponding scenario from the SQLite database
3. Retrieves vibration measurements related to that scenario
4. Generates a scenario summary
5. Generates rule-based maintenance recommendations
6. Returns a structured response with reasoning steps

## Tools Used by the Agent

The current agent orchestrates these project components:

```text
src.database.queries.get_scenario_by_id
src.database.queries.get_measurements_by_scenario
src.analysis.scenario_summary.generate_scenario_summaries
src.prescription.rule_based_recommendations.generate_recommendations
```

## Agent Response Structure

For a valid scenario, the agent returns:

```python
{
    "status": "success",
    "message": "Maintenance agent executed successfully.",
    "scenario_id": 1,
    "scenario": {...},
    "measurements_count": 2,
    "summary": {...},
    "recommendations": [...],
    "agent_reasoning": [...]
}
```

For an invalid scenario, the agent returns:

```python
{
    "status": "error",
    "message": "Scenario ID 999 not found.",
    "scenario_id": 999,
    "summary": None,
    "recommendations": [],
    "agent_reasoning": [...]
}
```

## Why the Agent is Deterministic First

The project intentionally starts with a deterministic agent before introducing real LLM behavior.

This provides several advantages:

* Easier debugging
* Lower cost
* Better testability
* Clearer execution flow
* No hallucinated tool calls
* Safer orchestration
* Better foundation for future LLM-based agents

The current design follows one important principle:

```text
Build deterministic orchestration first.
Add LLM reasoning only after the workflow is stable and testable.
```

## Current Capabilities

The deterministic maintenance agent can:

* Retrieve industrial scenario metadata
* Retrieve vibration measurements by scenario
* Summarize scenario-level diagnostic behavior
* Generate maintenance recommendations
* Flag cases requiring human validation
* Return transparent reasoning steps
* Support Streamlit visualization
* Support automated unit tests

## Current Limitations

The current agent does not yet:

* Use real LLM reasoning
* Select tools dynamically
* Call a RAG retriever
* Interpret free-text user instructions
* Compare multiple assets over time
* Produce source-grounded technical explanations
* Maintain long-running memory
* Execute multi-step planning

These limitations are intentional at this stage.

## Testing

The agent is covered by unit tests in:

```text
tests/test_maintenance_agent.py
```

The tests validate that the agent:

* Returns `success` for valid scenarios
* Returns `error` for invalid scenarios
* Returns scenario metadata
* Returns scenario summary
* Returns recommendations as structured records
* Returns agent reasoning

Run the tests with:

```bash
pytest tests/test_maintenance_agent.py
```

Or run the full test suite:

```bash
pytest
```

## Planned Evolution

Future versions of the agent may include:

### 1. Tool Registry

Create a registry of available tools, such as:

* SQL query tool
* Scenario summary tool
* Recommendation tool
* RAG retrieval tool
* ML prediction tool
* Human validation tool

### 2. Intent Routing

Add an intent router to decide which tools should be called based on the user request.

Example:

```text
"Show anomaly risk by measurement"
    → SQL query tool

"Explain lubrication degradation"
    → RAG retrieval tool

"What action should be evaluated?"
    → Recommendation tool + RAG tool

"Validate this diagnosis"
    → Human validation tool
```

### 3. RAG Integration

Connect the agent to a curated technical knowledge base.

The agent should be able to retrieve references about:

* Lubrication degradation
* Broadband spectral elevation
* Carpet patterns
* Structural looseness
* Harmonics and subharmonics
* Maintenance inspection procedures

### 4. Real LLM Tool Calling

After deterministic workflows are stable, the agent can be expanded with real LLM tool calling.

Required safeguards:

* Domain Guard before LLM calls
* Tool allowlist
* Structured tool inputs
* SQL Guard after SQL generation
* RAG source citation
* Human validation for corrective actions

### 5. Execution Trace

Future agent responses should include a detailed execution trace:

```python
{
    "tool": "get_measurements_by_scenario",
    "status": "success",
    "input": {"scenario_name": "structural_looseness"},
    "output_summary": "2 measurements retrieved"
}
```

### 6. Human-in-the-Loop Escalation

The agent should escalate high-risk cases when:

* Anomaly score is high
* Anomaly probability is high
* Scenario severity is medium or high
* Recommendation requires human validation
* Diagnostic confidence is uncertain

## Design Principle

The agent should remain:

```text
safe
testable
explainable
modular
cost-aware
human-supervised
```

The goal is not to create an autonomous industrial controller.

The goal is to create an explainable AI assistant that supports maintenance analysis and decision-making while keeping humans responsible for operational actions.