# Hybrid AI for Predictive Maintenance

A practical learning project combining Machine Learning, LLMs, Agentic AI and RAG for industrial vibration analysis.

## Project Overview

This project demonstrates a hybrid AI architecture for industrial predictive maintenance.

The use case is based on vibration analysis from rotating equipment. Historical vibration measurements are stored in a SQLite database and later consumed by analytical functions, ML models, natural language queries, agentic workflows and a technical RAG layer.

The project starts after data ingestion. The goal is to demonstrate how industrial data can be queried, analyzed, interpreted and converted into predictive and prescriptive recommendations.

## Core Idea

The project follows a clear separation of responsibilities:

- Machine Learning detects patterns, anomalies and likely fault conditions.
- LLMs make data interaction easier through natural language.
- Agents coordinate tools, route user intentions and manage analytical workflows.
- RAG retrieves curated technical references to support maintenance recommendations.
- Humans validate operational decisions before any corrective action is assumed.

## Industrial Scenario

The project simulates three vibration conditions:

1. Normal operation
2. Carpet pattern associated with possible lubrication issues
3. Structural looseness

The SQLite database is pre-populated with historical measurements and spectral features for each condition.

## Learning Goals

This project is designed as a learning path for applied industrial AI.

The main learning goals are:

1. Build a structured industrial data foundation using SQLite
2. Explore vibration data and spectral features
3. Compare recent behavior against historical baselines
4. Apply ML models for anomaly detection and fault classification
5. Use LLMs for Text-to-SQL interaction
6. Build agents that coordinate analytical tools
7. Create a technical RAG layer from curated maintenance references
8. Generate prescriptive recommendations with human validation
9. Discuss governance, traceability and operational risk in AI systems

## Architecture

```text
SQLite database
    ↓
Streamlit app
    ↓
Scenario selection
    ↓
Comparative analysis
    ↓
ML diagnosis
    ↓
Text-to-SQL
    ↓
Agentic workflow
    ↓
Technical RAG
    ↓
Recommendation
    ↓
Human validation
