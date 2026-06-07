# Embedding-Based Semantic Router

## Purpose

The embedding-based semantic router adds a semantic matching layer to the Hybrid AI Predictive Maintenance project.

Instead of relying only on exact keyword matching, the system uses sentence embeddings and cosine similarity to map natural language questions to approved predictive maintenance intents.

## Architecture

The routing flow is:

```text
User question
↓
Domain Guard
↓
Keyword Router
↓
Embedding Router fallback
↓
Approved SQL Template
↓
SQL Guard
↓
SQLite Database
````

## Model

The current implementation uses:

```text
SentenceTransformer: all-MiniLM-L6-v2
Similarity metric: cosine similarity
```

The model is used only to select the closest approved intent. It does not generate SQL.

## Governance Principle

The embedding model is not allowed to create database queries.

It only maps a user question to a predefined intent. The SQL executed by the system comes from approved templates and is still validated by the SQL Guard.

## Example

User question:

```text
which machines look most critical?
```

The embedding router maps it to:

```text
Intent: highest_risk_assets
Matched example: which machines look most critical
```

Then the system executes the approved SQL template associated with `highest_risk_assets`.

## Why not use an LLM for SQL?

This project intentionally avoids unrestricted LLM-based SQL generation.

The safer design is:

```text
Embeddings route.
SQL templates control.
LLM explains.
Agent coordinates.
Human validates.
```

## Future Agent Layer

A future AI maintenance agent can consume:

* selected intent
* SQL result
* similarity score
* matched semantic example
* diagnostic metrics
* human validation flags

The agent can then generate operational explanations and maintenance recommendations without controlling database access.