# Database Schema

This document describes the SQLite database used by the Hybrid AI for Predictive Maintenance project.

The database is designed to support deterministic analytics, semantic query routing, SQL templates, diagnostic exploration, and maintenance recommendations.

## Purpose

The database stores simulated industrial maintenance data, including:

- equipment information
- vibration measurements
- diagnostic scenarios
- maintenance recommendations
- human validation feedback

The schema supports a hybrid architecture where deterministic rules, SQL templates, and AI-assisted interfaces work together without allowing unrestricted SQL generation.

## Main Tables

## equipment

Stores basic information about monitored industrial assets.

Expected fields may include:

| Column | Description |
|---|---|
| equipment_id | Unique identifier of the equipment |
| equipment_name | Human-readable equipment name |
| equipment_type | Type or category of the equipment |
| location | Operational area or plant location |
| criticality | Business or operational criticality level |

## measurements

Stores vibration or condition-monitoring measurements collected from equipment.

Expected fields may include:

| Column | Description |
|---|---|
| measurement_id | Unique identifier of the measurement |
| equipment_id | Related equipment identifier |
| timestamp | Measurement timestamp |
| rms_velocity | Overall vibration velocity indicator |
| peak_amplitude | Peak vibration amplitude |
| dominant_frequency | Main frequency component |
| scenario_label | Diagnostic label associated with the signal |

## diagnostic_scenarios

Stores known or simulated diagnostic scenarios.

Expected fields may include:

| Column | Description |
|---|---|
| scenario_id | Unique identifier of the scenario |
| scenario_label | Name of the diagnostic scenario |
| description | Technical explanation of the scenario |
| severity | Severity level |
| likely_failure_mode | Related failure mode |

## recommendations

Stores deterministic maintenance recommendations generated from diagnostic scenarios.

Expected fields may include:

| Column | Description |
|---|---|
| recommendation_id | Unique identifier of the recommendation |
| measurement_id | Related measurement |
| scenario_label | Diagnostic scenario |
| priority | Recommended priority |
| recommended_action | Suggested maintenance action |
| technical_reason | Explanation behind the recommendation |
| human_validation_required | Indicates whether expert validation is required |

## human_validation

Stores feedback from human experts.

Expected fields may include:

| Column | Description |
|---|---|
| validation_id | Unique identifier of the validation |
| measurement_id | Related measurement |
| user_feedback | Human feedback or decision |
| validation_status | Accepted, rejected, or needs review |
| created_at | Feedback timestamp |

## Query Design Principles

The database is not exposed to free-form SQL generation.

Instead, the project uses:

- a Domain Guard to check whether the question belongs to the maintenance domain
- a Semantic Query Router to map questions to approved templates
- SQL templates for controlled database access
- a SQL Guard to prevent unsafe or unsupported queries

## Supported Query Patterns

The current query layer is designed to support questions such as:

- Which measurements are associated with a specific diagnostic scenario?
- Which equipment shows the highest vibration indicators?
- What recommendations are available for a given measurement?
- Which cases require human validation?
- What diagnostic scenarios exist in the database?

## Role in the Hybrid AI Architecture

This schema is a core part of the hybrid AI flow.

The database provides structured evidence.

The deterministic layer controls access.

The maintenance agent transforms structured diagnostic information into actionable recommendations.

Human validation remains required for maintenance decisions that may affect real operations.

## Future Improvements

Possible schema extensions:

- add sensor metadata
- add historical maintenance orders
- add failure history
- add spare parts information
- add technician notes
- add document references for future RAG integration
- add confidence scores for diagnostic scenarios
- add timestamps for recommendation generation