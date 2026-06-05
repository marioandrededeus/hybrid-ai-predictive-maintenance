# Hybrid AI for Predictive Maintenance

A practical learning project that explores how Machine Learning, deterministic agents, semantic query routing, safe Text-to-SQL, RAG and human validation can work together in an industrial predictive maintenance context.

The project focuses on vibration analysis from rotating equipment and demonstrates how industrial data can be queried, diagnosed, explained and converted into maintenance recommendations.

## Project Overview

This project demonstrates a hybrid AI architecture for industrial predictive maintenance.

The use case is based on vibration analysis from rotating equipment. Historical vibration measurements, spectral features, ML diagnostics and human feedback records are stored in a pre-populated SQLite database and consumed by a Streamlit application.

The project starts after data ingestion. Its purpose is to show how industrial data can be explored, queried, compared, diagnosed, explained and transformed into operational recommendations.

The current implementation already includes:

- SQLite data layer
- Streamlit scenario explorer
- Scenario summaries
- Rule-based maintenance recommendations
- Human validation workflow
- Safe mock Text-to-SQL workflow
- Domain Guard for prompt validation
- Semantic Query Router with predefined SQL templates
- SQL Guard for read-only query validation
- Simple deterministic maintenance agent
- Unit tests for guardrails and routing logic

Future versions will expand the project with real LLM integration, embedding-based retrieval, technical RAG and more advanced agentic orchestration.

## Core Idea

The project follows a clear separation of responsibilities:

- Machine Learning supports anomaly detection and diagnostic classification.
- Deterministic logic handles known, repetitive and safety-sensitive workflows.
- Semantic routing maps common questions to safe SQL templates.
- LLMs are reserved for cases where natural language reasoning is truly needed.
- Agents coordinate tools, route user intentions and manage analytical workflows.
- RAG will retrieve curated technical references to support maintenance recommendations.
- Humans validate operational decisions before corrective actions are assumed.

The main principle is simple:

```text
Do not use LLMs for everything.
Use deterministic and safe components first.
Use LLMs only when they add real value.
````

## Industrial Scenario

The project simulates three vibration conditions:

1. Normal operation
2. Carpet pattern associated with possible lubrication issues
3. Structural looseness

The SQLite database is pre-populated with historical measurements, spectral features and ML diagnostic outputs for each condition.

The focus is not real-time sensor generation. The focus is the analytical layer that consumes industrial data and turns it into insight, diagnosis and recommendation.

## Learning Goals

This project is designed as a practical learning path for applied industrial AI.

The main learning goals are:

1. Build a structured industrial data foundation using SQLite
2. Explore vibration data and spectral features
3. Summarize industrial scenarios from historical data
4. Apply diagnostic logic for anomaly and fault interpretation
5. Use controlled Text-to-SQL interaction
6. Add guardrails before any LLM-based workflow
7. Build agents that coordinate analytical tools
8. Create a technical RAG layer from curated maintenance references
9. Generate prescriptive recommendations with human validation
10. Discuss governance, traceability and operational risk in industrial AI systems

## Current Architecture

```text
SQLite Database
    ↓
Streamlit App
    ↓
Scenario Selection
    ↓
Diagnostic Data Exploration
    ↓
Scenario Summary
    ↓
Rule-Based Recommendations
    ↓
Human Validation
```

The current application also includes a controlled database assistant:

```text
User Prompt
    ↓
Domain Guard
    ↓
Semantic Query Router
    ↓
Predefined SQL Template or Mock Text-to-SQL
    ↓
SQL Guard
    ↓
SQLite Database
    ↓
Streamlit Result
```

A simple deterministic maintenance agent is also available:

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
```

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
  Scenario summary and analytical helpers

src/ml/
  ML model training and prediction layer

src/llm/
  Domain guard, semantic query router, SQL guard and Text-to-SQL utilities

src/agents/
  Agent routing, tool calling and orchestration

src/rag/
  Document ingestion, embeddings and retrieval

src/prescription/
  Maintenance recommendation logic

src/monitoring/
  Human feedback, validation records and traceability

knowledge_base/
  Curated technical references for the future RAG layer

notebooks/
  Learning notebooks for each project stage

docs/
  Architecture notes and technical documentation

tests/
  Unit tests for guardrails, routing and safety layers
```

## Hybrid Query Flow

The **Ask the Database** section uses a controlled hybrid query flow designed to avoid unnecessary LLM usage and keep database access safe.

Instead of sending every user question directly to a language model, the system first applies deterministic validation and routing layers:

```text
User Prompt
    ↓
Domain Guard
    ↓
Semantic Query Router
    ↓
Predefined SQL Template or Mock Text-to-SQL
    ↓
SQL Guard
    ↓
SQLite Database
    ↓
Streamlit Result
```

This flow includes:

* **Domain Guard**: blocks prompts outside the industrial predictive maintenance context.
* **Semantic Query Router**: maps known questions to predefined safe SQL templates.
* **Mock Text-to-SQL fallback**: handles valid domain questions that do not match a predefined template yet.
* **SQL Guard**: allows only safe read-only SQL queries and blocks destructive operations.
* **Execution path transparency**: shows whether the answer came from the router, fallback or was blocked.

This design supports a practical hybrid AI approach: use deterministic logic when possible, reserve LLM usage for cases where it is truly needed, and keep the system explainable, safe and cost-aware.

More details are available in:

```text
docs/hybrid_query_flow.md
```

## Supported Database Questions

The semantic router currently supports questions such as:

```text
Show average anomaly score by scenario
```

```text
Show average anomaly probability by scenario
```

```text
Which scenario has the highest risk?
```

```text
Which measurements require human validation?
```

```text
Show average RMS velocity by scenario
```

```text
Show lubrication issues
```

```text
Show structural looseness cases
```

```text
Which assets are monitored?
```

```text
Show anomaly risk by measurement
```

Questions outside the project context are blocked before reaching the SQL layer.

Example blocked prompt:

```text
Vibration level of Etna vulcan
```

Even though this prompt contains the word `vibration`, it is not related to industrial predictive maintenance.

## Maintenance Agent

The project includes a first deterministic maintenance agent.

The current agent does not use a real LLM yet. It orchestrates existing project tools:

* Retrieves the selected industrial scenario
* Retrieves vibration measurements related to the scenario
* Generates a technical scenario summary
* Applies rule-based maintenance recommendation logic
* Returns structured operational reasoning

Current flow:

```text
Scenario ID
    ↓
Maintenance Agent
    ↓
Database Query
    ↓
Scenario Summary
    ↓
Rule-Based Recommendations
    ↓
Agent Reasoning
```

This creates the foundation for future agentic workflows involving ML tools, RAG retrieval and human validation.

## Safety and Governance

The project includes safety and governance principles from the beginning:

* Prompts outside the industrial predictive maintenance domain are blocked.
* Known database questions are routed to predefined SQL templates.
* Only read-only SQL queries are allowed.
* Unsafe SQL commands are blocked before execution.
* Query execution path is displayed to the user.
* Rule-based recommendations are transparent.
* Recommendations requiring attention are flagged for human validation.
* The system does not assume direct control over industrial equipment.

Blocked SQL operations include:

```text
CREATE
ALTER
DROP
DELETE
UPDATE
INSERT
```

## Testing

The project includes unit tests for the main guardrail layers:

* Domain Guard
* Semantic Query Router
* SQL Guard

Run all tests with:

```bash
pytest
```

Example test coverage:

```text
Out-of-domain prompts are blocked
Known prompts are routed to SQL templates
Unknown but valid domain prompts fall back safely
Unsafe SQL commands are blocked
Safe SELECT queries are allowed
```

## Learning Path

### Phase 0: Problem Framing and Architecture

Define the industrial use case, the technical architecture and the learning path.

Deliverables:

* README
* Architecture notes
* Roadmap
* Project folder structure

### Phase 1: Industrial Data Foundation

Create a pre-populated SQLite database with industrial vibration scenarios.

Deliverables:

* SQLite database
* Asset table
* Scenario table
* Vibration measurements table
* Spectral features table
* ML diagnostics table
* Human feedback table
* Seed script

### Phase 2: Streamlit Data Explorer

Build an interface to select scenarios, inspect measurements and visualize trends.

Deliverables:

* Scenario selector
* KPI cards
* Measurement table
* Feature summary
* Diagnostic charts

### Phase 3: Scenario Summary

Generate technical summaries for each industrial scenario.

Deliverables:

* Scenario-level aggregation
* Risk level summary
* Technical interpretation helper

### Phase 4: Rule-Based Recommendations

Generate transparent maintenance recommendations based on diagnostic outputs.

Deliverables:

* Recommendation engine
* Priority classification
* Human validation flag

### Phase 5: Human Validation

Create a simple human-in-the-loop workflow.

Deliverables:

* Validation form
* Specialist notes
* Corrective action field
* Human feedback history

### Phase 6: Safe Database Assistant

Allow users to query the SQLite database using natural language with guardrails.

Deliverables:

* Domain Guard
* Semantic Query Router
* SQL templates
* Mock Text-to-SQL fallback
* SQL Guard
* Execution path transparency

### Phase 7: Deterministic Maintenance Agent

Create a simple agent that coordinates existing tools.

Deliverables:

* Agent orchestration function
* Scenario retrieval
* Measurement retrieval
* Scenario summary
* Recommendations
* Agent reasoning

### Phase 8: ML Diagnosis Expansion

Expand the ML layer for anomaly detection and fault classification.

Planned deliverables:

* Training script
* Saved model
* Prediction function
* Model evaluation
* Risk level classification

### Phase 9: Technical RAG

Create a curated technical knowledge base for maintenance recommendations.

Planned deliverables:

* Document ingestion
* Chunking
* Embeddings
* Vector database
* Retriever function
* Source-supported explanations

### Phase 10: Real LLM and Agentic Workflow

Introduce real LLM usage only after deterministic routing and safety layers.

Planned deliverables:

* LLM-based Text-to-SQL fallback
* Tool-calling agent
* RAG-supported recommendation generation
* Action logs
* Governance notes

### Phase 11: Deployment and Governance

Deploy the application and document limitations.

Planned deliverables:

* Streamlit deployment
* Secrets management
* Logging
* README update
* Technical limitations section

## Example Use Cases

The application is designed to support practical questions such as:

```text
Which scenario has the highest anomaly score?
```

```text
Which measurements require human validation?
```

```text
Is the selected scenario associated with lubrication degradation?
```

```text
Which cases indicate structural looseness?
```

```text
Which monitored assets are available in the database?
```

```text
What maintenance recommendations are generated for high-risk measurements?
```

## Important Design Principles

Each AI component has a specific role in the architecture.

Machine Learning supports analytical diagnosis.

Deterministic logic handles known and safety-sensitive workflows.

Semantic routing avoids unnecessary LLM calls.

LLMs support natural language interaction and explanation when deterministic logic is not enough.

Agents coordinate the workflow and decide which tools should be called.

RAG provides technical grounding for maintenance recommendations.

Operational decisions remain subject to human validation.

## Current Status

Implemented:

* SQLite database creation and seeding
* Streamlit scenario explorer
* Scenario summaries
* Rule-based recommendations
* Human validation workflow
* Safe mock Text-to-SQL flow
* Domain Guard
* Semantic Query Router
* SQL Guard
* Execution path transparency
* Deterministic maintenance agent
* Unit tests for guardrail layers
* Hybrid query flow documentation

Planned:

* Expanded ML training workflow
* Embedding-based semantic matching
* Vector database for known questions and SQL templates
* Technical RAG layer
* Real LLM-based Text-to-SQL fallback
* More advanced agentic orchestration
* Deployment notes

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Create and seed the database:

```bash
python src/database/create_database.py
python src/database/seed_database.py
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest
```

## Limitations

This project is a learning and demonstration environment.

Current limitations:

* The database is pre-populated and synthetic.
* The Text-to-SQL fallback is still mock-based.
* The maintenance agent is deterministic.
* The RAG layer is planned but not fully implemented yet.
* The system does not connect to real-time industrial sensors.
* The system must not be used to control industrial equipment directly.

## License

This project is intended for educational and portfolio purposes.