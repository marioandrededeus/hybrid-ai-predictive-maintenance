# Hybrid AI Predictive Maintenance

A Streamlit lab for industrial predictive maintenance that combines raw vibration signals, physics-informed anomaly scoring, and a controlled **Semantic AI** query layer.

This project was motivated by a practical discussion on **ML vs LLMs in manufacturing**: in industrial environments, decisions need to be traceable, auditable, cost-aware, and safe. The goal here is not to use a generative model as the source of operational truth. Instead, the app demonstrates how far a controlled semantic layer can go using embeddings, cosine similarity, keyword routing, approved SQL templates, and SQL validation, without depending on a paid GenAI API.

The guiding idea is:

```text
Physics-informed diagnostics decide.
Semantic AI routes and retrieves.
Deterministic rules recommend.
Humans remain responsible for operational validation.
```

> Note: this version does **not** implement an autonomous agent and does **not** implement a human-validation workflow in the application. Those concepts are part of the broader industrial AI positioning and future roadmap, but the current app focuses on condition monitoring, deterministic recommendations, and safe semantic querying.

---

## Live App

Add your deployed Streamlit URL here after publication:

```text
https://your-streamlit-app-url](https://semantic-ai-predictive-maintenance.streamlit.app
```

---

## Repository

```text
https://github.com/marioandrededeus/hybrid-ai-predictive-maintenance
```

---

## Why this project exists

A recent reflection on manufacturing AI can be summarized as:

```text
ML decides -> LLM explains -> Agent coordinates -> Human validates
```

However, in a public demo or cost-sensitive industrial context, using an external generative model for every interaction is not always necessary or desirable.

This project adapts that philosophy into a controlled, zero-cost semantic architecture:

- the diagnostic core is based on vibration signals and physics-informed features;
- the query experience uses semantic similarity instead of free-form LLM generation;
- the database is accessed only through approved SQL templates;
- SQL execution is protected by a SQL Guard;
- recommendations are deterministic and traceable;
- generative LLM usage is treated as an optional future layer, not a dependency.

This makes the app especially relevant for industrial scenarios where **cost, safety, auditability, and scope control** are design constraints.

---

## What the app demonstrates

The app has two main flows.

### 1. Monitoring

The Monitoring view shows the latest equipment condition up to a selected collection date.

Pipeline:

```text
selected collection date
-> latest measurement per equipment
-> raw vibration signal
-> time-domain metrics
-> FFT / PSD visualization
-> spectral features
-> deterministic anomaly score
-> severity classification
-> rule-based action plan
```

The current diagnostic layer is not a trained ML model yet. It is a **physics-informed deterministic baseline** derived from vibration features.

The app currently simulates and stores:

- raw vibration samples;
- RMS and peak velocity;
- FFT-derived features;
- broadband energy;
- low, mid, and high-frequency energy;
- harmonic and subharmonic indicators;
- anomaly scores;
- predicted condition;
- severity level;
- deterministic recommended action.

### 2. Semantic AI Query

The Semantic AI Query view allows users to ask maintenance-related questions in natural language-like form, but without giving a generative model unrestricted control.

Pipeline:

```text
user question
-> Domain Guard
-> Keyword Router
-> Embedding Router
-> cosine similarity
-> approved SQL template
-> SQL Guard
-> SQLite query
-> equipment cards
```

This design creates a guided natural-language experience with:

- no external LLM API cost;
- no private API key exposure;
- no arbitrary SQL generation;
- clear query execution trace;
- similarity scores for transparency;
- deterministic database access.

---

## Semantic AI instead of GenAI dependency

The project deliberately uses **Semantic AI** rather than a paid generative dependency in the current version.

Here, Semantic AI means:

```text
embeddings + cosine similarity + domain filters + approved templates + SQL safety
```

This is more limited than a full LLM-based assistant, but that limitation is intentional. The trade-off is valuable for a public industrial demo:

| Aspect | Semantic AI approach in this app |
|---|---|
| Cost | No external GenAI inference cost |
| Safety | Only approved SQL templates are executed |
| Scope | Questions must be related to predictive maintenance |
| Auditability | Routes, templates, and similarity scores are visible |
| Flexibility | More limited than free-form LLM generation |
| Industrial fit | Strong for controlled exploration of operational data |

This is not positioned as a replacement for LLMs. It is a pragmatic baseline showing that many useful industrial query experiences can be built with controlled semantic routing before introducing generative models.

---

## Relation with the previous condition monitoring project

This repository was motivated by a previous vibration analysis project:

```text
predictive-maintenance-vibration-lab
```

That earlier project explored vibration fault patterns such as:

- normal operation;
- carpet patterns associated with possible lubrication issues;
- structural looseness patterns.

The current project extends that idea into a broader application architecture:

```text
condition monitoring
+ raw signal storage
+ deterministic anomaly scoring
+ equipment cards
+ semantic database query
+ SQL safety
+ Streamlit deployment
```

The objective is to move from a signal-analysis lab to a more complete industrial decision-support demo.

---

## Current diagnostic logic

The anomaly score is calculated during the database seeding process from simulated raw vibration signals.

Conceptual flow:

```text
raw vibration signal
-> time-domain metrics
-> FFT / spectral features
-> carpet_score
-> looseness_score
-> overall_anomaly_score
-> anomaly_probability-like score
-> predicted_condition
-> severity
```

The main scoring patterns are:

### Carpet / lubrication pattern

The app considers features such as:

- broadband energy;
- high-frequency energy;
- spectral floor behavior;
- PSD-style energy spread.

This represents a simplified deterministic approximation of a carpet-like vibration pattern associated with possible lubrication degradation.

### Structural looseness pattern

The app considers features such as:

- low-frequency energy;
- harmonic ratio;
- subharmonic ratio;
- components around rotational frequencies such as 0.5x, 1x, 1.5x, 2x, and 3x.

This represents a simplified deterministic approximation of structural looseness behavior.

### Final anomaly score

Conceptually:

```text
overall_anomaly_score = max(carpet_score, looseness_score)
```

The final severity is threshold-based.

Important clarification:

> The current `anomaly_probability` should be interpreted as a normalized confidence-like score derived from deterministic rules, not as a calibrated probability from a trained classifier.

---

## What is implemented today

Implemented:

- SQLite database with raw vibration samples;
- synthetic data generation for multiple assets and scenarios;
- time-domain vibration metrics;
- FFT-based spectral features;
- PSD-style visualization;
- deterministic anomaly scoring;
- severity classification;
- rule-based recommended action plan;
- Streamlit monitoring dashboard;
- date-based monitoring filter;
- Semantic AI Query interface;
- Domain Guard;
- keyword-based routing;
- embedding-based semantic routing;
- cosine similarity matching;
- approved SQL templates;
- SQL Guard;
- query execution summary;
- project documentation inside the app footer.

Not implemented in the current version:

- external LLM API calls;
- free-form LLM-generated SQL;
- autonomous agents;
- human-validation workflow inside the app;
- trained ML anomaly detector;
- autoencoder, CNN, or LSTM models;
- model monitoring or MLOps pipeline.

---

## Architecture

High-level architecture:

```text
+-------------------------+
| Raw vibration simulator |
+------------+------------+
             |
             v
+-------------------------+
| SQLite database         |
| - assets                |
| - scenarios             |
| - vibration samples     |
| - measurements          |
| - spectral features     |
| - diagnostics           |
+------------+------------+
             |
             v
+-------------------------+
| Streamlit app           |
| - Monitoring            |
| - Semantic AI Query     |
+------------+------------+
             |
     +-------+--------+
     |                |
     v                v
+------------+   +----------------+
| Signal UI  |   | Semantic Query |
| FFT / PSD  |   | Guard + Router |
+------------+   +----------------+
```

---

## Project structure

```text
hybrid-ai-predictive-maintenance/
|
|-- app/
|   |-- streamlit_app.py
|   |-- page_icon.png
|   |-- background_sidebar.png
|
|-- data/
|   |-- hybrid_ai_predictive_maintenance.sqlite3
|
|-- docs/
|   |-- anomaly_scoring.md
|   |-- project_positioning.md
|
|-- src/
|   |-- database/
|   |   |-- create_database.py
|   |   |-- seed_database.py
|   |   |-- queries.py
|   |
|   |-- llm/
|   |   |-- domain_guard.py
|   |   |-- semantic_query_router.py
|   |   |-- embedding_router.py
|   |   |-- sql_guard.py
|   |   |-- text_to_sql.py
|   |
|   |-- agents/
|   |-- analysis/
|   |-- features/
|   |-- ml/
|   |-- prescription/
|   |-- rag/
|   |-- simulation/
|
|-- tests/
|-- requirements.txt
|-- README.md
```

Some folders are intentionally prepared for future expansion.

---

## How to run locally

### 1. Create and activate a virtual environment

Windows CMD:

```cmd
python -m venv hybrid_ai_st_env
hybrid_ai_st_env\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

### 3. Create and seed the database

```cmd
python -m src.database.create_database
python -m src.database.seed_database
```

### 4. Run tests

```cmd
pytest
```

### 5. Run the app

```cmd
streamlit run app\streamlit_app.py
```

---

## Example Semantic AI questions

Examples of supported natural-language-style questions:

```text
Which assets have high risk?
Show diagnostics with critical severity.
What is the average anomaly score by scenario?
Which equipment has the highest anomaly probability?
Show assets with possible lubrication issues.
Show structural looseness cases.
```

The system does not answer arbitrary questions. Unsupported or out-of-domain prompts are blocked or redirected to approved suggestions.

---

## Why not use a generative LLM in this version?

A generative LLM could be useful in future versions for:

- explaining asset condition in natural language;
- summarizing diagnostic evidence;
- generating SQL candidates under strict validation;
- supporting agentic maintenance workflows;
- preparing human-readable maintenance reports.

However, for a public app using a private API key, generative inference introduces practical concerns:

- API cost;
- token usage limits;
- key exposure risk;
- prompt injection risk;
- uncontrolled SQL generation risk;
- unnecessary complexity for a first production-style demo.

Therefore, this version intentionally demonstrates a lower-risk alternative:

```text
semantic retrieval and routing without external generative inference
```

The result is less flexible than a full LLM assistant, but more controlled, auditable, and cost-free to run.

---

## Roadmap

Potential next steps:

1. Refactor scoring logic out of `seed_database.py` into dedicated feature and scoring modules.
2. Add a trained anomaly detection baseline, such as Isolation Forest or One-Class SVM.
3. Add autoencoder-based anomaly detection for raw vibration windows.
4. Explore CNN or LSTM models for sequence-based fault detection.
5. Add optional LLM explanations behind a feature flag and daily token budget.
6. Add optional Text-to-SQL generation under stricter SQL Guard validation.
7. Add a real human-validation workflow for maintenance feedback.
8. Add agentic orchestration only after the diagnostic and validation layers are mature.

---

## Positioning statement

This project is not a claim that Semantic AI replaces LLMs or machine learning.

It demonstrates a more specific idea:

> In industrial contexts where cost, safety, and auditability matter, a controlled semantic layer using embeddings, cosine similarity, domain filters, and approved SQL templates can provide useful natural-language-style interaction without requiring paid generative inference.

The broader architecture remains aligned with the industrial AI thesis:

```text
Physics-informed and ML models should provide the decision core.
Semantic or language interfaces should make the system easier to query and understand.
Agents, when added, should coordinate workflows rather than replace engineering validation.
Humans remain accountable for operational decisions.
```
