# Project Roadmap

This roadmap tracks the implementation progress of the **Hybrid AI for Predictive Maintenance** project.

The project is designed as a practical learning path for combining deterministic analytics, Machine Learning, LLMs, agents, RAG and human validation in an industrial predictive maintenance context.

## Implemented

### 1. Project Foundation

- Repository structure
- Python environment setup
- `requirements.txt`
- `.gitignore`
- `.env.example`
- Initial documentation

### 2. SQLite Data Layer

- Database creation script
- Database seeding script
- Pre-populated SQLite database
- Query utilities
- Industrial assets table
- Maintenance scenarios table
- Vibration measurements table
- Spectral features table
- ML diagnostics table
- Human feedback table

### 3. Streamlit Application

- Main Streamlit app
- Scenario selector
- KPI cards
- Diagnostic charts
- Scenario interpretation
- Diagnostic data table

### 4. Scenario Summary

- Scenario-level aggregation
- Average anomaly score
- Average anomaly probability
- RMS velocity summary
- Spectral feature summary
- Risk level classification
- Technical interpretation helper

### 5. Rule-Based Recommendations

- Transparent recommendation logic
- Priority classification
- Technical reason generation
- Human validation flag

### 6. Human Validation Workflow

- Human validation form
- Specialist name
- Validated label
- Feedback notes
- Action required
- Human feedback history

### 7. Safe Database Assistant

- Mock Text-to-SQL workflow
- SQL validation
- Query execution
- Result display in Streamlit

### 8. Domain Guard

- Prompt validation before SQL or LLM workflows
- Out-of-domain prompt blocking
- Industrial predictive maintenance scope validation

### 9. Semantic Query Router

- Predefined SQL templates for known questions
- Safe routing before fallback
- Supported examples loaded dynamically in Streamlit

Current supported query intents include:

- Average anomaly score by scenario
- Average anomaly probability by scenario
- Highest risk scenarios
- Measurements requiring human validation
- Average RMS velocity by scenario
- Lubrication issues
- Structural looseness cases
- Monitored assets
- Anomaly risk by measurement

### 10. SQL Guard

- Read-only SQL validation
- Blocking of unsafe operations:
  - `CREATE`
  - `ALTER`
  - `DROP`
  - `DELETE`
  - `UPDATE`
  - `INSERT`

### 11. Execution Path Transparency

The Streamlit app shows whether a database question was:

- Blocked by the Domain Guard
- Answered by the Semantic Query Router
- Sent to the mock Text-to-SQL fallback
- Validated by the SQL Guard

### 12. Deterministic Maintenance Agent

- Scenario retrieval
- Measurement retrieval
- Scenario summary
- Rule-based recommendations
- Agent reasoning
- No real LLM usage yet

### 13. Tests

Unit tests were added for:

- Domain Guard
- Semantic Query Router
- SQL Guard

## In Progress

### 1. Documentation

Current documentation includes:

- Main README
- Hybrid query flow document
- Project roadmap

Next documentation improvements:

- Architecture diagram
- Data schema documentation
- Agent design notes
- RAG design notes

### 2. Query Layer Maturity

The current query layer already includes guardrails and routing.

Next improvements:

- Add more SQL templates
- Add template metadata
- Add template descriptions
- Add confidence score for routing
- Add tests for SQL template execution

## Next Steps

### 1. Add More Semantic Query Templates

Potential new intents:

- Highest anomaly measurements
- Average temperature by scenario
- Average load by scenario
- Peak velocity by scenario
- Diagnostics by predicted label
- Measurements by asset
- Recommendations by priority
- Human validation queue

### 2. Improve Semantic Routing

Possible improvements:

- Add simple scoring instead of keyword matching only
- Add synonyms per intent
- Add route confidence
- Add fallback rules
- Add clear messages for unsupported questions

### 3. Add SQL Template Execution Tests

Create tests that validate:

- Each template returns a valid SQL query
- Each template passes SQL Guard
- Each template can execute against the SQLite database
- Each template returns expected columns

### 4. Improve Maintenance Agent Output

Current agent output can be improved by:

- Returning fully serializable responses
- Converting DataFrames to records
- Adding status per tool call
- Adding execution trace
- Adding error handling for missing data
- Adding tests for agent behavior

### 5. Add Technical Documentation

Suggested files:

```text
docs/architecture.md
docs/database_schema.md
docs/agent_design.md
docs/rag_design.md
docs/testing_strategy.md
````

## Future Evolution

### 1. Embedding-Based Semantic Matching

Replace or complement keyword routing with embeddings.

Possible approach:

* Store canonical questions and SQL templates
* Generate embeddings for user prompts
* Compare with cosine similarity
* Use SQL template when similarity score is above threshold
* Use LLM fallback only when similarity is low

Suggested threshold for first experiment:

```text
similarity >= 0.78 → use known SQL template
similarity < 0.78 → fallback
```

### 2. Real LLM-Based Text-to-SQL

After deterministic guardrails are stable, introduce real LLM-based Text-to-SQL.

Required controls:

* Domain Guard before LLM call
* Schema-aware prompt
* Allowed tables and columns
* SQL Guard after generation
* Read-only enforcement
* Query limit enforcement
* Logging of generated SQL

### 3. Technical RAG Layer

Create a curated technical knowledge base for maintenance reasoning.

Planned components:

* Raw technical documents
* Text chunking
* Embeddings
* Vector store
* Retriever
* Source-supported explanations
* Recommendation grounding

### 4. Advanced Agentic Workflow

Future agent capabilities:

* Route user intent
* Decide whether to query SQL, ML, RAG or recommendation tools
* Combine diagnostic data with technical references
* Generate explainable maintenance recommendations
* Escalate high-risk cases to human validation

### 5. ML Layer Expansion

Future ML improvements:

* Train anomaly detection models
* Compare rule-based diagnostics with ML predictions
* Add model evaluation
* Save trained models
* Add prediction pipeline
* Add feature importance or explainability

### 6. Deployment and Governance

Planned deployment and governance topics:

* Streamlit deployment
* Secrets management
* Logging
* Monitoring
* Usage limits
* Model limitations
* Human validation policy
* Safety disclaimer

## Design Principle

The project follows one central principle:

```text
Use deterministic, safe and explainable components first.
Use LLMs only when they add value.
Keep humans in the loop for operational decisions.
```