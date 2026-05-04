# Meta-Agent — Risk Register

**Project:** AI Tutoring System — Final Year Research Project
**Component:** Meta-Agent
**Document version:** 1.0
**Status:** Draft for Progress Presentation 1

---

## 1. Purpose

This document identifies risks to the successful delivery of the Meta-Agent component, assesses their likelihood and impact, and defines mitigation strategies. Risks are reviewed and updated at each project milestone.

---

## 2. Risk scoring

**Likelihood:** Low (1) · Medium (2) · High (3)
**Impact:** Low (1) · Medium (2) · High (3)
**Score:** Likelihood × Impact (1–9)
**Priority:** 1–2 Low · 3–4 Medium · 6–9 High

---

## 3. Risk register

### R1 — Concept extraction inaccuracy

| Field | Detail |
|-------|--------|
| **Category** | Technical / ML |
| **Description** | The prompted LLM may misidentify concepts or misclassify mastery signals, leading to an inaccurate knowledge graph |
| **Likelihood** | High (3) |
| **Impact** | High (3) |
| **Score** | 9 |
| **Priority** | High |
| **Mitigation** | Use structured JSON output with strict schema validation; design prompts with concrete few-shot examples; spot-check extraction outputs against ground-truth transcripts manually; plan a fine-tuned classifier as a future enhancement |
| **Owner** | Component owner |
| **Status** | Active — mitigation in progress |

---

### R2 — BKT model underperforms on real data

| Field | Detail |
|-------|--------|
| **Category** | Technical / ML |
| **Description** | The trained BKT model may produce inaccurate mastery estimates if the ASSISTments data distribution differs significantly from the system's actual usage patterns |
| **Likelihood** | Medium (2) |
| **Impact** | Medium (2) |
| **Score** | 4 |
| **Priority** | Medium |
| **Mitigation** | Evaluate on held-out test split with AUC and RMSE; benchmark against published results on the same dataset; document model limitations clearly; design the system to allow model swap (BKT → DKT) without architectural change |
| **Owner** | Component owner |
| **Status** | Active — evaluation pipeline being built |

---

### R3 — Integration delays with teammates

| Field | Detail |
|-------|--------|
| **Category** | Project / Coordination |
| **Description** | Teammates working on the multi-agent framework, self-improving agent, or memory component may not be ready to integrate within the timeline, blocking end-to-end testing |
| **Likelihood** | Medium (2) |
| **Impact** | Medium (2) |
| **Score** | 4 |
| **Priority** | Medium |
| **Mitigation** | Publish API contract early so teammates can develop against a fixed interface; provide mock endpoints returning realistic data so others are unblocked; agree on a transcript JSON schema with the memory component owner before week 1 ends |
| **Owner** | Component owner |
| **Status** | Active — API contract being drafted |

---

### R4 — LLM API availability and cost

| Field | Detail |
|-------|--------|
| **Category** | External dependency |
| **Description** | Concept extraction depends on an LLM API; outages, rate limits, or unexpected costs could disrupt development or demonstration |
| **Likelihood** | Low (1) |
| **Impact** | Medium (2) |
| **Score** | 2 |
| **Priority** | Low |
| **Mitigation** | Cache extraction results during development to avoid repeated calls; use a smaller/cheaper model where acceptable; have a fallback rule-based extractor for demo continuity |
| **Owner** | Component owner |
| **Status** | Monitoring |

---

### R5 — Cold-start profiling unreliable

| Field | Detail |
|-------|--------|
| **Category** | Technical / Design |
| **Description** | First-message linguistic analysis for new students may produce profiles that are too noisy to be useful, especially for short or unusual first messages |
| **Likelihood** | Medium (2) |
| **Impact** | Low (1) |
| **Score** | 2 |
| **Priority** | Low |
| **Mitigation** | Treat the cold-start profile as a soft prior, not a hard label; replace with empirical data after the first complete session; allow tutor agents to override the profile if it conflicts with observed behaviour |
| **Owner** | Component owner |
| **Status** | Active — mitigation built into design |

---

### R6 — Scope creep across team boundaries

| Field | Detail |
|-------|--------|
| **Category** | Project / Coordination |
| **Description** | The Meta-Agent's responsibilities can blur into those of the memory component (storage) or self-improving agent (reflection), leading to duplicated or missing work |
| **Likelihood** | Medium (2) |
| **Impact** | Medium (2) |
| **Score** | 4 |
| **Priority** | Medium |
| **Mitigation** | Clear scope boundaries documented in requirements (Section 2 of `requirements.md`); regular team check-ins to confirm component ownership; explicit handoff points defined in the API contract |
| **Owner** | Project team (shared) |
| **Status** | Active — boundaries documented |

---

### R7 — Demo failure during progress presentation

| Field | Detail |
|-------|--------|
| **Category** | Project / Delivery |
| **Description** | Live demo may fail due to environment issues, API errors, or unexpected input, undermining the proof-of-concept evidence |
| **Likelihood** | Medium (2) |
| **Impact** | High (3) |
| **Score** | 6 |
| **Priority** | High |
| **Mitigation** | Prepare a pre-recorded video as backup; rehearse the demo at least three times in the actual presentation environment; cache API responses for the demo flow; prepare static screenshots for each demo step as a final fallback |
| **Owner** | Component owner |
| **Status** | Planned — to be addressed in week 2 |

---

### R8 — Insufficient evaluation evidence

| Field | Detail |
|-------|--------|
| **Category** | Academic / Rubric |
| **Description** | Without quantitative evaluation, the project may be marked down on technology demonstration and proof of concept |
| **Likelihood** | Low (1) |
| **Impact** | High (3) |
| **Score** | 3 |
| **Priority** | Medium |
| **Mitigation** | Produce a held-out test evaluation with AUC and RMSE for the BKT model; produce manual accuracy assessment of concept extraction on a small labelled sample; document evaluation methodology explicitly |
| **Owner** | Component owner |
| **Status** | Active — evaluation pipeline planned |

---

### R9 — Knowledge graph schema rigidity

| Field | Detail |
|-------|--------|
| **Category** | Technical / Design |
| **Description** | A schema designed only for math may need significant rework if the project pivots to a multi-subject scope |
| **Likelihood** | Low (1) |
| **Impact** | Medium (2) |
| **Score** | 2 |
| **Priority** | Low |
| **Mitigation** | Design schema with a generic `concept_id` and `domain` field from the start; avoid math-specific fields in the core schema; document the assumption in the requirements |
| **Owner** | Component owner |
| **Status** | Mitigated by design |

---

### R10 — Team member unavailability

| Field | Detail |
|-------|--------|
| **Category** | Project / People |
| **Description** | Illness, exam pressure, or other commitments may reduce available development time over the two-week window |
| **Likelihood** | Medium (2) |
| **Impact** | Medium (2) |
| **Score** | 4 |
| **Priority** | Medium |
| **Mitigation** | Front-load critical path work (BKT training, API skeleton) into week 1; build buffer time into week 2; ensure documentation is current so work can be picked up by another team member if needed |
| **Owner** | Component owner |
| **Status** | Monitoring |

---

## 4. Summary table

| ID  | Risk                                       | Score | Priority |
|-----|--------------------------------------------|-------|----------|
| R1  | Concept extraction inaccuracy              | 9     | High     |
| R7  | Demo failure during progress presentation  | 6     | High     |
| R2  | BKT model underperforms on real data       | 4     | Medium   |
| R3  | Integration delays with teammates          | 4     | Medium   |
| R6  | Scope creep across team boundaries         | 4     | Medium   |
| R10 | Team member unavailability                 | 4     | Medium   |
| R8  | Insufficient evaluation evidence           | 3     | Medium   |
| R4  | LLM API availability and cost              | 2     | Low      |
| R5  | Cold-start profiling unreliable            | 2     | Low      |
| R9  | Knowledge graph schema rigidity            | 2     | Low      |

---

## 5. Review cadence

This risk register is reviewed:
- At each weekly team meeting
- After every progress presentation
- When any new dependency or integration point is added
