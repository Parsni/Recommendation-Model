# Multi-Criteria LLM Selection & Contextual Routing Engine
## Enterprise System Architecture, Database Schema, & Mathematical Blueprint

This document details the complete end-to-end implementation of the **Multi-Criteria LLM Selection & Contextual Routing Engine**, trained directly on the pairwise comparison dataset and architected to meet rigorous software architecture and principal engineering standards.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    Client["Client App / API Gateway / Front-End"] --> Layer1["1. INGESTION LAYER<br/>Request Sanitizer & Bounds Evaluator"]
    Layer1 --> Layer2["2. IN-FLIGHT FEATURIZATION ENGINE<br/>(Tokenization, Length, Syntactic Density, Language)"]
    
    Layer2 --> TrackA["Track A: Unsupervised Clustering<br/>(K-Means Intent Segmentation)"]
    Layer2 --> TrackB["Track B: Cosine Similarity<br/>(Production Anchor Baselines)"]
    Layer2 --> TrackC["Track C: Metadata Extractor<br/>(Structural & User Metrics)"]
    
    TrackA --> Bridge["Context Enrichment Bridge<br/>(High-Density Tabular Vector)"]
    TrackB --> Bridge
    TrackC --> Bridge
    
    Bridge --> Layer3["3. COGNITIVE INFERENCE LAYER"]
    Layer3 --> LogReg["Logistic Regression Ensemble<br/>Predicts P(Success)"]
    Layer3 --> RidgeReg["Dual-Headed Ridge Regressors<br/>Forecasts Cost ($) & Latency (ms)"]
    
    LogReg --> Layer4["4. SCORING MATRIX & COMPOSITE ROUTER<br/>Bounded Linear Min-Max Scaling & Weighted Optimization"]
    RidgeReg --> Layer4
    
    Layer4 --> Layer5["5. RECONCILIATION ENGINE & DISPATCH<br/>Failsafe Gate & Policy Failover Routing"]
    
    Layer5 --> LLMTarget["Target Destination LLM API<br/>(GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama 3.1 70B)"]
    Layer5 --> Layer6["6. TELEMETRY & FEEDBACK LOOP<br/>(PostgreSQL / SQLAlchemy Storage & Ground Truth Retraining)"]
    LLMTarget -.-> Layer6
```

---

## 2. Mathematical Formulations & ML Pipeline

### 2.1 Track A & B: Spatial Featurization
- **Intent Clustering ($K$-Means)**: Inbound prompt $x$ is vectorized via TF-IDF $\phi_{\text{tfidf}}(x) \in \mathbb{R}^{d}$. Centroids $c_1, \dots, c_k$ assign an Intent Cluster ID:
  $$\text{Cluster}(x) = \arg\min_{j} \|\phi_{\text{tfidf}}(x) - c_j\|_2$$
- **Production Anchor Proximity (Cosine Similarity)**: Given verified enterprise baseline strings $A = \{a_1, \dots, a_M\}$:
  $$\text{Sim}(x, a_j) = \frac{\phi_{\text{tfidf}}(x) \cdot \phi_{\text{tfidf}}(a_j)}{\|\phi_{\text{tfidf}}(x)\|_2 \|\phi_{\text{tfidf}}(a_j)\|_2}$$
  $$\text{MaxSim}(x) = \max_{j} \text{Sim}(x, a_j) \in [0.0, 1.0]$$

### 2.2 Cognitive Inference: Bradley-Terry Logistic Regression Ensemble
Given pairwise competition records $(x, m_A, m_B)$ where $y \in \{0, 1\}$ denotes whether model $A$ beat model $B$:
- Latent contextual utility of candidate model $m$ on prompt feature vector $\mathbf{z}(x)$:
  $$U(m, x) = \mathbf{w}_m^T \mathbf{z}(x) + b_m$$
- Pairwise comparison probability:
  $$P(m_A \succ m_B \mid x) = \sigma(U(m_A, x) - U(m_B, x)) = \frac{1}{1 + e^{-(U(m_A, x) - U(m_B, x))}}$$
- Calibrated Absolute Success Probability:
  $$P_{\text{BT}}(\text{Success} \mid x, m) = \sigma(U(m, x) - \bar{U}(x)) \quad \text{where} \quad \bar{U}(x) = \frac{1}{|\mathcal{M}|}\sum_{k \in \mathcal{M}} U(k, x)$$
- Ensembled with regularized pointwise logistic classification and Laplace prior smoothing:
  $$P(\text{Success} \mid x, m) = \text{clip}\left(0.60 \cdot P_{\text{BT}} + 0.40 \cdot P_{\text{pointwise}}, \; 0.05, \; 0.98\right)$$

### 2.3 Resource Forecasting: Dual-Headed Ridge Regression
- **Computational Cost Regressor ($\hat{C}_m$)**:
  $$\hat{C}(m, x) = \boldsymbol{\alpha}_m^T \mathbf{f}_{\text{tokens}}(x) + c_m \quad \text{(USD)}$$
- **Network Execution Latency Regressor ($\hat{L}_m$)**:
  $$\hat{L}(m, x) = \boldsymbol{\gamma}_m^T \mathbf{f}_{\text{tokens}}(x) + l_m \quad \text{(ms)}$$
Both models trained with $L_2$ regularization objective: $\min_{\boldsymbol{\beta}} \|y - X\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2$.

### 2.4 Bounded Linear Min-Max Scaling & Composite Scoring
Heterogeneous dimensions (currency, milliseconds, probabilities) are normalized into strict scalar bounds $[0.0, 1.0]$. Lower costs and latencies are inverted to represent efficiency:
$$\text{CostEff}_m = \frac{\max_{k}(C_k) - C_m}{\max_k(C_k) - \min_k(C_k) + \epsilon}$$
$$\text{LatencyEff}_m = \frac{\max_{k}(L_k) - L_m}{\max_k(L_k) - \min_k(L_k) + \epsilon}$$
The Multi-Criteria Composite Score is computed as:
$$\text{Final Score}_m = (w_1 \cdot P_m(\text{Success})) + (w_2 \cdot \text{MaxSim}_m) + (w_3 \cdot \text{CostEff}_m) + (w_4 \cdot \text{LatencyEff}_m)$$
subject to $\sum_{i=1}^4 w_i = 1.0$.

### 2.5 Reconciliation & The Failsafe Gate
The highest-scoring model $m^* = \arg\max_m \text{Final Score}_m$ is evaluated by the Failsafe Gate:
- If $P(m^* \mid \text{Success}) < \tau_{\text{failsafe}}$ (default $0.40$) OR $\text{Final Score}_{m^*} < 0.35$:
  - Bypass $m^*$ and route request directly to the designated fallback engine (e.g. `gpt-4o`).
  - Flag `fallback_triggered = True` with structural audit justification.

---

## 3. Database Schema (PostgreSQL / SQLAlchemy)

The database schema is designed for multi-tenant high throughput, audit compliance, and continuous ground truth model updates.

### 3.1 Entity Relationship Diagram

```mermaid
erDiagram
    ROUTING_REQUESTS ||--o{ CANDIDATE_EVALUATIONS : "generates"
    ROUTING_REQUESTS ||--|| ROUTING_DECISIONS : "produces"
    ROUTING_DECISIONS ||--o| EXECUTION_TELEMETRY : "measures"
    ROUTING_DECISIONS ||--o| USER_FEEDBACK : "collects"

    ROUTING_REQUESTS {
        string id PK
        string user_id
        text sanitized_prompt
        int char_length
        int estimated_tokens
        string detected_language
        int intent_cluster_id
        string intent_cluster_label
        string nearest_anchor_id
        float anchor_similarity
        timestamp created_at
    }

    CANDIDATE_EVALUATIONS {
        string id PK
        string request_id FK
        string model_id
        int rank
        float composite_score
        float success_prob
        float similarity
        float predicted_cost_usd
        float predicted_latency_ms
        float cost_efficiency
        float latency_efficiency
        bool is_selected
    }

    ROUTING_DECISIONS {
        string id PK
        string request_id FK
        string selected_model
        bool fallback_triggered
        text fallback_reason
        float routing_latency_ms
        jsonb audit_metadata
        timestamp dispatched_at
    }

    EXECUTION_TELEMETRY {
        string id PK
        string decision_id FK
        string model_id
        float actual_latency_ms
        int actual_input_tokens
        int actual_output_tokens
        float actual_cost_usd
        int status_code
        text error_message
        timestamp recorded_at
    }

    USER_FEEDBACK {
        string id PK
        string decision_id FK
        string user_id
        int rating
        bool is_positive
        string preferred_winner_model
        text feedback_comment
        bool used_for_retraining
        timestamp created_at
    }

    ANCHOR_BASELINES {
        string id PK
        string category
        text prompt_text
        text description
        bool is_active
        timestamp created_at
    }

    ROUTING_POLICIES {
        string id PK
        string policy_name
        float w_success
        float w_similarity
        float w_cost
        float w_latency
        float failsafe_threshold
        bool is_active
        timestamp updated_at
    }
```

### 3.2 SQLAlchemy Implementation Reference
The complete SQLAlchemy ORM implementation is located at:
- [`src/routing/telemetry/db_models.py`](file:///d:/Recommendation%20Model/recommendation_engine/src/routing/telemetry/db_models.py)
- PostgreSQL DDL script: [`outputs/schema/postgresql_schema.sql`](file:///d:/Recommendation%20Model/recommendation_engine/outputs/schema/postgresql_schema.sql)

---

## 4. Empirical Training Results on Real Battle Data

The engine is trained on **3,945 real head-to-head preference battles** from
public LMArena-style arena logs (`data/routing/arena_battles_real.csv`; dataset
composition and provenance recorded in `data/routing/KNOWN_FACTS_ARENA.md`).
Each row contains a prompt, both candidates and the human-preferred winner;
intent clusters are assigned by the shared domain detector
(`src/routing/domains.py`). Training uses a prompt-grouped 80/20 split
(seed 42): 3,151 train / 794 held-out battles.

### Held-Out Accuracy (dual metric, honest)

| Metric | Definition | Result |
| :--- | :--- | :--- |
| **Metric A** – Task-Category Agreement | Share of held-out prompts where the router picks the model with the highest shrunk empirical win-rate in that prompt's task category (Beta(1,1) prior on the train split). Target ≥ 90%. | **92.82 %** ✔ (n = 794) |
| **Metric B** – Head-to-Head Preference Agreement | Share of held-out battles where the router's composite score ranks the human winner above the loser (chance ≈ 50 %). | **54.28 %** |
| 5-fold grouped CV (stability check) | Same two metrics over five prompt-grouped folds | A = 88.08 % ± 4.77, B = 55.26 % ± 0.92 |

Hyper-parameters (regularization C, Bradley-Terry blend, domain-affinity
scale) were selected by a logged grid search over the train split only
(`scripts/tune_accuracy.py`, results in `outputs/accuracy/tuning_results.json`);
the final values live in `config/routing_config.json` (`inference` block).

### Empirical Win Rates (train split, Beta(1,1) shrunk)

| Model | Games | Shrunk win rate |
| :--- | ---: | ---: |
| deepseek-r1 | 189 | 60.2 % |
| gpt-4o | 1,843 | 56.3 % |
| llama-3.1-70b | 476 | 51.0 % |
| qwen3-235b-a22b | 322 | 49.4 % |
| gpt-4o-mini | 488 | 48.6 % |
| gemini-1.5-pro | 1,378 | 47.1 % |
| claude-3-5-sonnet | 1,606 | 44.3 % |

Empirical best model per task category (Metric A ground truth):
math → qwen3-235b-a22b, code → gpt-4o, creative → gemini-1.5-pro,
translation → gemini-1.5-pro, doc → deepseek-r1, comm → deepseek-r1,
qa → qwen3-235b-a22b, general → deepseek-r1.

Rebuild the report at any time with
`python scripts/build_accuracy_report.py --with-cv`.

---

## 5. UI/UX Operational Guide

Launch the dedicated Streamlit dashboard:
```bash
streamlit run app_router.py
```

### 5.1 Experience levels (dual-audience design)

A sidebar radio switches the whole app between three depths — one app for
laymen and evaluators alike (`src/routing/ui_panels.py`):

- **👶 Simple** – plain-language results only: the "In everyday terms" card
  (task in everyday words, cost/speed traffic-light chips vs roster medians),
  plain accuracy line, compact "Who are these 7 chatbots?" roster cards, and
  the glossary. All ML internals hidden.
- **Standard** (default) – everything above plus the **Under the Hood**
  section: per-model ensemble internals table (latent utility U(m,x),
  σ(U−Ū) Bradley-Terry prob, pointwise prob, blended P + 95% CI, domain
  affinity, train win-rate), the **What-If Sandbox** (4 weight sliders +
  failsafe-threshold slider → deterministic instant re-rank of the cached
  predictions with a Δ-rank delta table), the learned domain-affinity
  heatmap (models × domains, detected domain highlighted), the feature
  inspector (structural features, domain one-hot, K-Means cluster, nearest
  anchor), cost/latency model economics, failsafe/audit trail, decision JSON
  export (download + API-style payload preview), and a model-roster spec
  table.
- **🔧 Expert** – everything above plus hyperparameters & provenance
  (C, bt_blend, affinity scale, min games, weights, seed, dataset path,
  train/test counts, unrounded metrics) and the local-SQLite telemetry
  panel (recent persisted decisions, unprocessed feedback).

The layman card, plain accuracy line, preset captions ("Best all-rounder…",
"Optimise for the lowest cost…") and glossary are always available; nothing
numeric is rounded up in any mode.

### 5.2 Guided flow

The app runs as eight guided steps with a **persistent decision trail**
(every route decision recorded in-session; entries survive tab switches):

1. **Prompt & templates** – 10 everyday task chips plus free text.
2. **Optimization preset** – 6 strategies with plain-English captions.
3. **Buffering pipeline** – animated 6-stage trail, persisted at the end.
4. **Winner reveal** – composite score, justification chips, plus the
   "In everyday terms" plain-summary card and traffic-light cost/speed/
   certainty chips.
5. **Accuracy (honest dual metric)** – Metric A / Metric B hero numbers
   (2-decimal render), plain-English one-liner, per-domain agreement table,
   confusion matrix, win-rate heatmap, cost-latency scatter; definitions,
   sample counts and the 5-fold CV stability line are always shown; targets
   are never rounded up.
6. **Decision Analytics + Under the Hood** – success/composite bars,
   cost-latency scatter, weighted-contribution breakdown chart (quality /
   similarity / cost / speed segments summing to the composite), then the
   Standard/Expert panels described in 5.1.
7. **Comparison deck** – all 7 ranked candidate cards with real Arena
   win-rates; Simple shows compact roster cards, Standard/Expert a full
   spec table.
8. **Feedback page** – star rating, alternative-winner selection and
   comments persisted to the feedback store, plus a retraining expander
   (open via *Continue to Feedback →*).

A fully **offline rule-based chat assistant** (`src/routing/chat_assistant.py`)
answers help / why / compare / switch-policy / metrics / feedback / status /
new-prompt intents anywhere in the app — no API keys, no network calls.
Ground-truth feedback and alternative winners append to the battle dataset
and trigger engine retraining.

New charts (`src/routing/charts.py`): `chart_affinity_heatmap` (signed
centered-logit matrix with the detected domain outlined) and
`chart_contribution_breakdown` (stacked weighted contributions per
candidate).

---

## 6. Verification & Test Suite Summary

- **Total Test Suite**: 141 tests passing (103 core + prompt/generic/routing-engine
  regression tests + 38: 8 dual-metric accuracy, 9 offline chat assistant,
  9 `app_router` AppTest flows incl. experience-level gating and sandbox
  re-ranking, 12 `ui_panels` pure-helper unit tests, chart renderers incl.
  the two new charts).
- **How to run**: `python -m pytest tests -v` (log: `outputs/logs/pytest_verbose.txt`,
  summary: `docs/test_results.md`).
- **Browser walkthrough**: headed Playwright (Edge) 28/28 checks, 0 console
  errors — Standard/Expert/Simple modes, sandbox cost-only re-rank (cheapest
  model promoted to #1), screenshots in `outputs/screenshots/walkthrough/v4_*`.
- **Headline verified numbers**: Metric A = 92.82 % (held-out, target ≥ 90 %),
  Metric B = 54.28 %, 5-fold CV A = 88.08 % ± 4.77 — all reproduced by
  `python scripts/build_accuracy_report.py --with-cv`.
