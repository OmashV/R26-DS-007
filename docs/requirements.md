# Meta-Agent — Requirements Document

**Project:** AI Tutoring System — Final Year Research Project
**Component:** Meta-Agent (Persistent Student Modelling & Learning Path Generation)
**Document version:** 1.0
**Status:** Draft for Progress Presentation 1

---

## 1. Purpose

The Meta-Agent is the persistent student-modelling component of a multi-agent AI tutoring system. While the live tutoring agents handle real-time conversation, the Meta-Agent operates between sessions to build, maintain, and update a structured model of each student's knowledge over time. It produces a personalised learning path that evolves as the student interacts with the system.

This document specifies the user, functional, and non-functional requirements for the Meta-Agent component.

---

## 2. Scope

### 2.1 In scope

- Post-session analysis of tutoring conversation transcripts
- Extraction of concepts and mastery signals from unstructured dialogue
- Per-student knowledge graph construction and maintenance
- Knowledge regression detection across sessions
- Personalised learning path generation
- First-message linguistic analysis for new-student cold-start profiling
- REST API for integration with other components in the tutoring system

### 2.2 Out of scope

- Live in-session tutoring (handled by the multi-agent framework)
- Real-time response evaluation (handled by the self-improving agent)
- Raw transcript storage (handled by the memory component)
- Content generation (lessons, problems, explanations)

### 2.3 Domain assumption

The initial implementation targets **K-12 mathematics** as the working domain, since the ASSISTments dataset used for training the knowledge tracing model covers this domain. The architecture is subject-agnostic; extending to other domains requires retraining the BKT model on a domain-appropriate dataset.

---

## 3. Stakeholders and users

### 3.1 Primary users
- **Students (K-12)** — receive personalised learning paths via the tutoring interface

### 3.2 Secondary users (system-internal)
- **Tutor agents** — query student profiles to adapt explanations
- **Self-improving agent** — uses misconception flags to trigger reflection
- **Memory component** — supplies session transcripts for analysis

---

## 4. User requirements

| ID  | Requirement |
|-----|-------------|
| UR1 | A student shall receive a learning path indicating what to revise, what to learn next, and what they have already mastered |
| UR2 | A student's profile shall persist across sessions so the system remembers them |
| UR3 | A student new to the system shall receive an appropriately calibrated starting profile within their first interaction |
| UR4 | A student's profile shall update automatically after each session without manual intervention |
| UR5 | A student whose mastery of a previously known concept declines shall have that concept flagged for revision |

---

## 5. Functional requirements

### 5.1 Transcript ingestion

| ID   | Requirement |
|------|-------------|
| FR1  | The system shall accept a session transcript in a defined JSON format via REST API |
| FR2  | The system shall validate transcript structure before processing and reject malformed input |
| FR3  | The system shall associate each transcript with a unique student identifier |

### 5.2 Concept extraction

| ID   | Requirement |
|------|-------------|
| FR4  | The system shall identify mathematical concepts present in a transcript |
| FR5  | The system shall classify each identified concept by mastery signal (strong, partial, weak) |
| FR6  | The system shall distinguish gaps in knowledge from active misconceptions |
| FR7  | The system shall produce concept extraction output as structured JSON consumable by downstream components |

### 5.3 Knowledge tracing

| ID   | Requirement |
|------|-------------|
| FR8  | The system shall maintain a probability estimate of mastery for each (student, concept) pair |
| FR9  | The system shall update mastery probabilities using a trained Bayesian Knowledge Tracing model |
| FR10 | The system shall classify a concept as mastered when the mastery probability exceeds a configurable threshold |

### 5.4 Knowledge graph

| ID   | Requirement |
|------|-------------|
| FR11 | The system shall persistently store a per-student concept map |
| FR12 | The system shall record a timestamped history of mastery state changes |
| FR13 | The system shall expose the current knowledge graph for a given student via the API |

### 5.5 Regression detection

| ID   | Requirement |
|------|-------------|
| FR14 | The system shall detect when a previously mastered concept's mastery probability declines below threshold |
| FR15 | The system shall flag regressed concepts for prioritisation in the learning path |

### 5.6 Learning path generation

| ID   | Requirement |
|------|-------------|
| FR16 | The system shall generate a personalised learning path containing three categories: revise urgently, learn next, already strong |
| FR17 | After resolved session evidence updates mastery, the cross-session pipeline shall regenerate the learning path as a deterministic derived view of current mastery and curriculum data |
| FR18 | The system shall expose the current learning path for a given student via the API |

### 5.7 New-student cold-start

| ID   | Requirement |
|------|-------------|
| FR19 | The system shall analyse a new student's first message to infer initial profile parameters (vocabulary level, support need, pace preference) |
| FR20 | The system shall replace the inferred profile with empirical data once the student completes their first full session |

---

## 6. Non-functional requirements

| ID    | Category        | Requirement |
|-------|-----------------|-------------|
| NFR1  | Performance     | Concept extraction shall complete within 30 seconds for a typical session transcript (under 50 turns) |
| NFR2  | Performance     | API responses for profile and learning path queries shall return within 500 ms under nominal load |
| NFR3  | Reliability     | The system shall persist all state changes such that no student progress is lost on restart |
| NFR4  | Maintainability | The system shall follow modular separation between API, core logic, ML model, and data access layers |
| NFR5  | Maintainability | All public functions shall include type hints and docstrings |
| NFR6  | Testability     | Each core module shall have unit test coverage for its primary code path |
| NFR7  | Privacy         | Student identifiers shall be stored as opaque IDs with no personally identifiable information in the knowledge graph |
| NFR8  | Portability     | The system shall run on any platform supporting Python 3.10+ |

---

## 7. API contract (high-level)

| Endpoint                          | Method | Purpose |
|-----------------------------------|--------|---------|
| `/session/process`                | POST   | Ingest a transcript and run the full update pipeline |
| `/student/{id}/profile`           | GET    | Retrieve current knowledge graph for a student |
| `/student/{id}/path`              | GET    | Retrieve current learning path for a student |
| `/student/new`                    | POST   | Initialise a new student profile from their first message |

A detailed API contract with request/response schemas is maintained in `docs/api_contract.md`.

---

## 8. Technology stack

| Layer              | Technology |
|--------------------|------------|
| API framework      | FastAPI |
| Knowledge tracing  | pyBKT (Bayesian Knowledge Tracing) |
| Concept extraction | LLM via API (prompted classification) |
| Persistent storage | SQLite (development); architecture supports migration to PostgreSQL |
| Training data      | ASSISTments 2009-2010 skill builder dataset |
| Language           | Python 3.10+ |

---

## 9. Assumptions and dependencies

### 9.1 Assumptions
- Session transcripts are made available by the memory component in a defined JSON format
- The LLM provider used for concept extraction is available and operational
- Initial scope is limited to mathematics; multi-domain support is future work

### 9.2 Dependencies
- **Memory component** — must supply session transcripts after each session
- **Multi-agent framework** — must invoke `/session/process` at session end
- **Tutor agents** — must consume profile and learning path data through the API

---

## 10. Acceptance criteria for Progress Presentation 1

| Criterion | Evidence |
|-----------|----------|
| BKT model trained and evaluated on a real dataset | Notebook with training pipeline, AUC and RMSE on held-out test set |
| Concept extraction working on sample transcripts | Demo input transcript producing structured concept output |
| Architecture and module structure implemented | Repository with documented folder structure and stub implementations |
| API endpoints reachable | At least `/session/process` and `/student/{id}/profile` returning valid responses |
| Integration interface defined for teammates | API contract document and mock responses available |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **BKT** | Bayesian Knowledge Tracing — a probabilistic model that estimates student mastery from sequences of correct/incorrect responses |
| **Concept graph** | A per-student data structure mapping concepts to mastery state |
| **Mastery signal** | A categorical or probabilistic indicator of how well a student understands a concept |
| **Misconception** | A specific incorrect belief held by a student, distinct from a knowledge gap |
| **Regression** | A decline in a student's mastery of a previously understood concept |
| **Cold-start** | The problem of generating useful recommendations for a new student with no history |
