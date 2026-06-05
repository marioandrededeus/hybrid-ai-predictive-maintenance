# Hybrid AI for Predictive Maintenance

A practical learning project combining Machine Learning, LLMs, Agentic AI and RAG for industrial vibration analysis.

## Project Overview

This project demonstrates a hybrid AI architecture for industrial predictive maintenance.

The use case is based on vibration analysis from rotating equipment. Historical vibration measurements are stored in a SQLite database and consumed by analytical functions, Machine Learning models, natural language queries, agentic workflows and a technical RAG layer.

The project starts after data ingestion. Its purpose is to show how industrial data can be queried, compared, diagnosed, explained and converted into operational recommendations with human validation.

## Core Idea

The project follows a clear separation of responsibilities:

- Machine Learning detects patterns, anomalies and likely fault conditions.
- LLMs make interaction with data easier through natural language.
- Agents coordinate tools, route user intentions and manage analytical workflows.
- RAG retrieves curated technical references to support maintenance recommendations.
- Humans validate operational decisions before corrective actions are assumed.

## Industrial Scenario

The project simulates three vibration conditions:

1. Normal operation
2. Carpet pattern associated with possible lubrication issues
3. Structural looseness

The SQLite database is pre-populated with historical measurements and spectral features for each condition.

The focus is not real-time sensor generation. The focus is the analytical layer that consumes industrial data and turns it into insight, diagnosis and recommendation.

## Learning Goals

This project is designed as a practical learning path for applied industrial AI.

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
Historical data exploration
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
Prescriptive recommendation
    ↓
Human validation
````

## Repository Structure

```text
app/
  Streamlit application

src/database/
  SQLite creation, seeding and query utilities

src/simulation/
  Synthetic vibration signal generation used to create the pre-populated database

src/features/
  Time-domain and frequency-domain feature extraction

src/analysis/
  Historical baseline and comparative analysis

src/ml/
  ML model training and prediction

src/llm/
  Text-to-SQL and LLM utilities

src/agents/
  Agent routing, tool calling and orchestration

src/rag/
  Document ingestion, embeddings and retrieval

src/prescription/
  Maintenance recommendation logic

src/monitoring/
  Logs, human feedback and traceability

knowledge_base/
  Curated technical references for the RAG layer

notebooks/
  Learning notebooks for each project stage

docs/
  Architecture, roadmap and teaching notes
```

## Learning Path

### Phase 0: Problem Framing and Architecture

Define the industrial use case, the technical architecture and the learning path.

Deliverables:

* README
* Architecture document
* Roadmap
* Teaching notes
* Project folder structure

### Phase 1: Industrial Data Foundation

Create a pre-populated SQLite database with industrial vibration scenarios.

Deliverables:

* SQLite database
* Asset table
* Vibration measurements table
* Spectral features table
* Seed script

### Phase 2: Streamlit Data Explorer

Build an interface to select scenarios, inspect measurements and visualize trends.

Deliverables:

* Scenario selector
* Measurement table
* Feature summary
* Trend visualization
* Signal visualization

### Phase 3: Comparative Analysis

Compare recent vibration behavior against historical baselines.

Example question:

```text
What is the main difference in the vibration spectrum from the last 2 hours compared to the last 30 days?
```

Deliverables:

* Last 2 hours vs last 30 days comparison
* Feature difference table
* Interpretation helper

### Phase 4: ML Diagnosis

Apply ML models to identify anomalies and likely fault conditions.

Deliverables:

* Training script
* Saved model
* Prediction function
* Risk level classification

### Phase 5: Text-to-SQL

Allow users to query the SQLite database using natural language.

Deliverables:

* Schema-aware prompt
* SQL generation
* SQL validation
* Query execution
* Result explanation

### Phase 6: Agentic Workflow

Create an agent that routes user requests and calls the appropriate tools.

Examples:

* Simple data question → Text-to-SQL
* Temporal comparison → Comparative analysis tool
* Diagnostic question → ML tool
* Corrective action question → ML + RAG + prescription tool

Deliverables:

* Intent router
* Tool registry
* Agent orchestrator
* Action logs

### Phase 7: Technical RAG

Create a curated technical knowledge base for maintenance recommendations.

Deliverables:

* Document ingestion
* Chunking
* Embeddings
* Vector database
* Retriever function

### Phase 8: Prescriptive Recommendation

Generate maintenance recommendations based on data, ML output and retrieved technical references.

Deliverables:

* Recommendation engine
* Source-supported explanation
* Human validation interface

### Phase 9: Deployment and Governance

Deploy the application and document limitations.

Deliverables:

* Streamlit deployment
* Secrets management
* Logging
* README update
* Technical limitations section

## Example User Questions

The application is designed to support questions such as:

```text
Which measurements have the highest carpet score?
```

```text
What is the main difference in the vibration spectrum from the last 2 hours compared to the last 30 days?
```

```text
Is the current asset showing signs of lubrication degradation?
```

```text
Which features are driving the anomaly?
```

```text
Considering the observed carpet risk, what corrective action should be evaluated?
```

## Important Design Principles

Each AI component has a specific role in the architecture.

Machine Learning remains responsible for analytical diagnosis.

LLMs support natural language interaction and explanation.

Agents coordinate the workflow and decide which tools should be called.

RAG provides technical grounding for maintenance recommendations.

Operational decisions remain subject to human validation.

## Safety and Governance

The project includes governance principles from the beginning:

* Only read-only SQL queries are allowed in the Text-to-SQL layer.
* Generated SQL must be validated before execution.
* Agent actions must be logged.
* ML predictions must be traceable.
* RAG sources must be stored with metadata.
* Corrective recommendations require human validation.
* The system should not assume direct control over industrial equipment.
