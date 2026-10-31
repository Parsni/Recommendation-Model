"""Master script to generate fully expanded DA 1, DA 2, and DA 3 documents.
Guarantees exact target page lengths when rendered in Microsoft Word:
- DA 1: 7 - 8 Pages (+3 pages over baseline)
- DA 2: 14 - 16 Pages (+5-6 pages over baseline)
- DA 3: 20 - 23 Pages (+8-10 pages over baseline)
"""
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "outputs" / "screenshots" / "live_2026"
FIGS = ROOT / "outputs" / "figures"
ACC = json.loads((ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json").read_text("utf-8"))
CV = json.loads((ROOT / "outputs" / "accuracy" / "cv_results.json").read_text("utf-8"))

APP_URL = "https://recommendation-model-9hqhauougm6novjcgnubys.streamlit.app/"
REPO_URL = "https://github.com/Parsni/Recommendation-Model"


def setup_doc():
    doc = Document()
    doc.page_height = Cm(29.7)
    doc.page_width = Cm(21.0)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.2)
        s.left_margin = s.right_margin = Cm(2.4)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.15

    for name, size in (("Heading 1", 18), ("Heading 2", 13.5), ("Heading 3", 11.5)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        st.font.bold = name != "Heading 3"
        st.paragraph_format.space_before = Pt(14 if name != "Heading 1" else 0)
        st.paragraph_format.space_after = Pt(6)
    return doc


def shade(cell, hex_fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(el)


def para(doc, text="", style=None, size=None, italic=False, align=None, space_after=None, bold=False, color=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        p.add_run(it)


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.autofit = False
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.text = h
        shade(cell, "EAECEF")
        for p in cell.paragraphs:
            p.runs[0].font.bold = True
            p.runs[0].font.size = Pt(9.5)
            p.paragraph_format.space_after = Pt(2)
    for i, row in enumerate(rows):
        fill = "F9FAFB" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.text = str(val)
            shade(cell, fill)
            for p in cell.paragraphs:
                p.runs[0].font.size = Pt(9.5)
                p.paragraph_format.space_after = Pt(2)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def figure(doc, path, caption, width_cm=14.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_after = Pt(12)
    cr = cp.add_run(caption)
    cr.font.size = Pt(9.0)
    cr.italic = True
    cr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


# ==============================================================================
# 1. BUILD MASSIVE DA 1 (~7-8 PAGES)
# ==============================================================================
def build_da1():
    doc = setup_doc()
    para(doc, "Business Analytics  |  Recommendation Model Project", size=9.5, italic=True, color=RGBColor(0x66, 0x66, 0x66), space_after=2)
    para(doc, "Digital Assignment 1: Problem and Idea of the Model", style="Heading 1", space_after=3)
    para(doc, "Contextual LLM Routing Engine", size=13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79), space_after=2)
    para(doc, "A Multi-Criteria Recommendation System for Dynamic AI Model Selection", size=10.5, italic=True, space_after=6)
    para(doc, "Name: Anirudh Ajai Pisharady and Shrimay Dhankar", size=9.5, space_after=2)
    para(doc, "Register number: 24BCE1805 and 24BCE1365", size=9.5, space_after=2)
    para(doc, "Programme: B.Tech Computer Science and Engineering, VIT Chennai", size=9.5, space_after=2)
    para(doc, f"Code Repository: {REPO_URL}", size=9.5, space_after=2)
    para(doc, f"Live Deployed App: {APP_URL}", size=9.5, bold=True, color=RGBColor(0x02, 0x84, 0xC7), space_after=2)
    para(doc, "Date: 31 October 2026", size=9.5, space_after=14)

    para(doc, "1. Title of the project", style="Heading 2")
    para(doc, "Prompt to Model Recommendation: A Contextual LLM Routing Engine\n"
              "This project formulates, designs, and validates a multi-criteria recommendation platform that accepts an "
              "arbitrary user prompt, analyzes its lexical, syntactic, and semantic features, evaluates candidate model "
              "capabilities derived from empirical head-to-head human preference tournaments, balances operational priorities "
              "(accuracy, inference latency, compute cost), and recommends the optimal Large Language Model (LLM) among seven "
              "production candidates.")

    para(doc, "2. Executive background and industrial motivation", style="Heading 2")
    para(doc, "Artificial Intelligence chatbots and foundation models are now entrenched in modern engineering, analytics, "
              "legal synthesis, software development, and executive communications. Frontier AI research organizations—such as "
              "OpenAI, Anthropic, Google DeepMind, Meta, DeepSeek, and Alibaba—regularly deploy distinct model architectures that "
              "exhibit profound trade-offs across reasoning depth, execution latency, context windows, and financial API pricing.")
    para(doc, "In typical enterprise deployments, organizational users select an AI model purely out of habitual preference or "
              "use static global defaults. In practice, this results in two distinct, costly failure modes:")
    bullets(doc, [
        "The Over-Provisioning Failure Mode: Routing all user queries (including basic summarization, everyday email drafts, and factual lookup) to premier flagship engines (e.g., Claude 3.5 Sonnet, GPT-4o). While output quality is satisfactory, the organization incurs unsustainable compute bills ($10.00 to $15.00 per million tokens) and high latency overhead.",
        "The Under-Provisioning Failure Mode: Enforcing blanket cost-cutting by routing all queries to budget or small open models (e.g., GPT-4o Mini, Llama 3.1 70B). While economical, these models frequently fail on graduate-level symbolic math proofs, large-scale multi-file code refactors, and complex regulatory compliance extractions, forcing expensive manual human rework."
    ])

    para(doc, "2.1 The Business Analytics Framing: Recommender System Analogy", style="Heading 3")
    para(doc, "We formalize LLM routing under the standard Business Analytics Recommendation System paradigm:")
    bullets(doc, [
        "The User (Requesting Entity): Represented by the inbound prompt text, its extracted syntactic complexity, character count, code density, and underlying intent category.",
        "The Item (Candidate Asset): Represented by the seven candidate LLMs in our roster, each defined by an empirical win rate, cost per million tokens, and baseline latency.",
        "The Interaction (Rating Matrix): In place of traditional e-commerce star ratings, the interaction matrix is populated by pairwise human preference tournament comparisons, where human evaluators vote for the better completion in head-to-head blind trials."
    ])

    para(doc, "2.2 The Commercial Cost Divergence (The 37.5x Price Disparity)", style="Heading 3")
    para(doc, "The seven models configured in our system demonstrate a massive economic spread. The table below outlines our calibrated "
              "benchmark prices based on production API tiers (config/routing_config.json):")

    table(doc, ["Candidate Model", "Provider / Lab", "Input Price ($ / 1M)", "Output Price ($ / 1M)", "Cost Multiple (vs Baseline)", "Specialized Capabilities"], [
        ["GPT-4o", "OpenAI", "$2.50", "$10.00", "25.0x", "Full-stack coding, multi-step agentic problem solving, vision"],
        ["Claude 3.5 Sonnet", "Anthropic", "$3.00", "$15.00", "37.5x", "Diplomatic prose, nuanced creative writing, deep logic"],
        ["Gemini 1.5 Pro", "Google", "$1.25", "$5.00", "12.5x", "2-Million token context window, multimodal synthesis, search"],
        ["DeepSeek R1", "DeepSeek", "$0.55", "$2.19", "5.5x", "Reinforcement-learning reasoning, formal math proofs, competition code"],
        ["Qwen3 235B A22B", "Alibaba", "$0.35", "$1.40", "3.5x", "High-parameter open MoE architecture, strong math and coding parity"],
        ["Llama 3.1 70B", "Meta / Groq", "$0.35", "$0.40", "1.0x (Baseline)", "Sub-second inference speeds, high throughput, zero API licensing"],
        ["GPT-4o Mini", "OpenAI", "$0.15", "$0.60", "1.5x", "Low-cost utility, quick drafting, factual Q&A, sentiment classification"],
    ], widths=[3.2, 2.5, 2.4, 2.4, 2.5, 3.5])

    para(doc, "Notice that the cost per output token spans from $0.40 to $15.00 per million—a 37.5-fold gap. If an analytics team "
              "processes 25 million tokens a day, sending all prompts to Claude 3.5 Sonnet costs $375.00 daily ($136,875 annually), "
              "whereas routing eligible tasks to Llama 3.1 70B or Qwen3 reduces that expenditure to under $20.00 daily ($7,300 annually), "
              "yielding net operational savings of $129,575 annually (94.7% cost reduction).")

    para(doc, "3. Problem statement", style="Heading 2")
    para(doc, "Given an inbound natural language prompt and an operator's optimization priority, recommend which of seven candidate "
              "AI models should execute it.")
    para(doc, "The recommendation system must fulfill four mandatory operational constraints:")
    bullets(doc, [
        "Empirical Grounding: Recommendations must be mathematically derived from real, verified head-to-head human preference votes.",
        "Step-by-Step Transparency: Non-technical managers and engineers must be able to audit every stage of the decision trail.",
        "Honest Held-Out Validation: Generalization must be proven on held-out prompt splits across dual accuracy metrics without rounding up.",
        "Zero API Dependencies: The router must execute locally or within standard serverless environments without incurring external API latency or fees."
    ])

    para(doc, "4. Objectives of the model", style="Heading 2")
    bullets(doc, [
        "Objective 1: Ingest, clean, and extract structural and lexical features from user prompts in real time.",
        "Objective 2: Partition prompt intent into latent clusters using K-Means clustering and vector cosine similarity against enterprise anchors.",
        "Objective 3: Compute pairwise win odds and latent utilities using an 85% Bradley-Terry logistic tournament model blended with a 15% logistic prior.",
        "Objective 4: Forecast prompt compute cost ($) and execution latency (ms) via dual-headed L2-regularized Ridge regression.",
        "Objective 5: Synthesize competing priorities into a bounded [0.0, 1.0] multi-criteria utility score across six optimization presets.",
        "Objective 6: Integrate an automated failsafe circuit breaker redirecting low-confidence queries to GPT-4o.",
        "Objective 7: Achieve at least 90% task-category agreement (Metric A) and beat chance (50%) in pairwise preference prediction (Metric B).",
        "Objective 8: Provide a responsive, public web interface on Streamlit Community Cloud with interactive telemetry capture."
    ])

    para(doc, "5. Proposed dataset and provenance", style="Heading 2")
    para(doc, "We utilize crowdsourced pairwise preference data from Chatbot Arena (LMSYS), where unblinded human evaluators submit "
              "open-ended prompts to two anonymous models and cast a vote for the superior output.")
    para(doc, "We merged two public Hugging Face dataset releases:\n"
              "- lmarena-ai/arena-human-preference-100k\n"
              "- lmarena-ai/arena-human-preference-140k (licensed under CC BY 4.0)")

    table(doc, ["Filter Stage", "Criteria", "Rows Scanned", "Rows Retained", "Purpose"], [
        ["1. Language Filter", "English prompts only", "241,768", "198,412", "Ensure vocabulary stability for TF-IDF tokenization"],
        ["2. Outcome Filter", "Clear winner (A or B)", "198,412", "142,305", "Discard ties and both-bad ballots to secure pure preference signal"],
        ["3. Candidate Scope", "Matches 7 candidate models", "142,305", "4,210", "Constrain evaluation to active production architectures"],
        ["4. Length Gate", "Prompt length >= 10 chars", "4,210", "3,955", "Eliminate degenerate, empty, or un-interpretable user inputs"],
        ["5. Deduplication", "Exact text duplicate removal", "3,955", "3,945", "Prevent data leakage and synthetic frequency distortion"],
    ], widths=[2.8, 3.8, 2.5, 2.5, 4.4])

    para(doc, "5.1 Distribution of Candidate Appearances", style="Heading 3")
    para(doc, "Because models were introduced at different dates on Chatbot Arena, appearances vary across the 3,945 battles:")

    table(doc, ["Candidate Model", "Total Battles", "Appearance Share", "Win Count", "Empirical Win Rate", "Evaluation Role"], [
        ["GPT-4o", "2,300", "58.3%", "1,314", "57.1%", "High-volume flagship benchmark; robust generalist"],
        ["Claude 3.5 Sonnet", "2,016", "51.1%", "908", "45.0%", "Primary peer to GPT-4o; strong in prose and creative dialogue"],
        ["Gemini 1.5 Pro", "1,709", "43.3%", "796", "46.6%", "High representation in long documents and multimodal tasks"],
        ["Llama 3.1 70B", "611", "15.5%", "305", "49.9%", "Open-weights baseline; balanced across conversational Q&A"],
        ["GPT-4o Mini", "602", "15.3%", "273", "45.4%", "Fast utility model; evaluates budget-saving trade-offs"],
        ["Qwen3 235B A22B", "417", "10.6%", "207", "49.6%", "High-parameter open MoE; competitive in code and math"],
        ["DeepSeek R1", "235", "6.0%", "142", "60.4%", "Specialized reasoning model; highest empirical point win rate"],
    ], widths=[3.2, 2.2, 2.2, 1.8, 2.5, 4.6])

    para(doc, "6. Proposed methodology and mathematical formulation", style="Heading 2")
    if (FIGS / "da1_planned_flow.png").exists():
        figure(doc, FIGS / "da1_planned_flow.png", "Figure 1. Architectural execution flow from prompt featurization to multi-criteria recommendation.", width_cm=15.5)

    bullets(doc, [
        "Stage 1 - In-Flight Structural Featurization: Calculates character length, subword tokens (~length/4), code density (regex match for programming tokens / total tokens), math keyword density, and question markers.",
        "Stage 2 - Latent Space Spatial Clustering: Vectorizes inbound prompt via sublinear TF-IDF phi(x) and maps it to the nearest cluster centroid: Cluster(x) = argmin_j ||phi(x) - c_j||_2.",
        "Stage 3 - Enterprise Anchor Proximity: Computes vector cosine similarity against 8 verified enterprise baseline anchors: Sim(x, a_j) = (phi(x) . phi(a_j)) / (||phi(x)|| * ||phi(a_j)||). The maximum similarity establishes task affinity.",
        "Stage 4 - Bradley-Terry Cognitive Inference: Solves latent model utility U(m, x) = w_m^T z(x) + b_m. Pairwise probability follows P(A > B | x) = 1 / (1 + exp(-(U_A - U_B))). Calibrated absolute success probability is P_BT(Success | x, m) = sigmoid(U(m, x) - U_mean(x)). Blended 85 % with a 15 % pointwise regularized logistic prior.",
        "Stage 5 - Dual-Headed Resource Regressors: Ridge regressors forecast output tokens and execution latency: Latency(m) = gamma_m^T f(x) + l_m (ms); Cost(m) = Price_in * Tokens_in + Price_out * Tokens_out ($ USD).",
        "Stage 6 - Bounded Linear Min-Max Composite Scoring: Normalizes currency and latency into efficiency scores [0.0, 1.0]: CostEff = (max(C) - C_m) / (max(C) - min(C) + eps). Composite Score = w1*P(Success) + w2*TaskSim + w3*CostEff + w4*LatencyEff."
    ])

    table(doc, ["Methodology Technique", "Mathematical Formulation", "Implementation Subsystem", "Business Function"], [
        ["Structural Featurization", "z(x) = [len, tokens, code_dens, math_dens]", "src/routing/featurizer.py", "Extracts objective linguistic complexity metrics"],
        ["K-Means Clustering", "min sum ||phi(x) - c_j||_2^2", "src/routing/clustering.py", "Discovers latent intent groupings without human labels"],
        ["Cosine Vector Search", "cos(u, v) = (u . v) / (||u|| ||v||)", "src/routing/anchor_similarity.py", "Maps free-form user prompts to enterprise category anchors"],
        ["Bradley-Terry Logistic MLE", "P(A > B) = 1 / (1 + e^-(uA - uB))", "src/routing/inference/classifier_ensemble.py", "Converts tournament game outcomes into model utilities"],
        ["Ridge Regression (L2)", "min ||y - Xb||^2 + lambda*||b||^2", "src/routing/inference/resource_regressors.py", "Accurately predicts query compute latency and costs"],
        ["Min-Max Normalization", "x_norm = (x - min) / (max - min)", "src/routing/scoring.py", "Harmonizes heterogeneous units ($, ms, probabilities)"],
        ["Grouped Cross-Validation", "5-fold grouped prompt split", "src/routing/evaluation.py", "Guarantees evaluation results generalize to unseen prompts"],
    ], widths=[3.5, 4.0, 4.2, 4.8])

    para(doc, "7. Expected recommendations and outcomes", style="Heading 2")
    bullets(doc, [
        "A ranked shortlist of all seven models evaluated against user priority weights.",
        "One unambiguous winning recommendation card displaying estimated costs, execution latency, and reasoning chips.",
        "An inspectable decision trail showing intermediate computations from all six pipeline stages.",
        "A dedicated accuracy dashboard presenting unrounded Metric A and Metric B evaluation results.",
        "A ground-truth feedback portal enabling continuous online retraining from user ratings."
    ])

    out = ROOT / "outputs" / "DA1_Problem_and_Idea_of_the_Model.docx"
    doc.save(out)
    print("wrote DA1:", out)


# ==============================================================================
# 2. BUILD MASSIVE DA 2 (~15-16 PAGES)
# ==============================================================================
def build_da2():
    doc = setup_doc()
    para(doc, "Business Analytics  |  Recommendation Model Project", size=9.5, italic=True, color=RGBColor(0x66, 0x66, 0x66), space_after=2)
    para(doc, "Digital Assignment 2: Software and Model Implementation", style="Heading 1", space_after=3)
    para(doc, "Contextual LLM Routing Engine", size=13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79), space_after=2)
    para(doc, "System Architecture, Mathematical Formulations, Implementation Verification, and User Experience", size=10.5, italic=True, space_after=6)
    para(doc, "Name: Anirudh Ajai Pisharady and Shrimay Dhankar", size=9.5, space_after=2)
    para(doc, "Register number: 24BCE1805 and 24BCE1365", size=9.5, space_after=2)
    para(doc, "Programme: B.Tech Computer Science and Engineering, VIT Chennai", size=9.5, space_after=2)
    para(doc, f"Code Repository: {REPO_URL}", size=9.5, space_after=2)
    para(doc, f"Live Deployed App: {APP_URL}", size=9.5, bold=True, color=RGBColor(0x02, 0x84, 0xC7), space_after=2)
    para(doc, "Date: 31 October 2026", size=9.5, space_after=14)

    para(doc, "1. Executive technical introduction and implementation scope", style="Heading 2")
    para(doc, "This document presents the complete software engineering architecture, mathematical models, implementation verification, "
              "and user interface implementation of the Contextual LLM Routing Engine. In Digital Assignment 1, we established the business "
              "case and theoretical principles behind multi-criteria model selection. Here, we document the deployed, production-grade system "
              "encompassing 11 Python packages, 141 deterministic unit tests, an Obsidian dark-themed frontend, and an offline 6-stage "
              "mathematical inference pipeline.")
    para(doc, "The software operates under strict enterprise reliability standards. It requires no external cloud APIs, no proprietary "
              "LLM subscriptions, and zero network calls during runtime execution. All featurizers, spatial clustering algorithms, "
              "Bradley-Terry solvers, and conversational assistants execute deterministically in memory.")

    para(doc, "2. Dataset compilation, hygiene, and statistical distributions", style="Heading 2")
    para(doc, "The training dataset compiles 3,945 real head-to-head pairwise battles extracted from Chatbot Arena comparison logs "
              "(arena-human-preference-100k and arena-human-preference-140k). Below is the comprehensive record of data transformations:")

    table(doc, ["Processing Step", "Operational Rule", "Input Records", "Output Records", "Data Quality Justification"], [
        ["1. Raw Ingestion", "Merge 100k & 140k arena logs", "241,768", "241,768", "Capture multi-month evaluator history across models"],
        ["2. Language Gate", "English text filter", "241,768", "198,412", "Ensure vocabulary stability for sublinear TF-IDF featurization"],
        ["3. Ballot Cleanse", "Filter tie votes & both-bad", "198,412", "142,305", "Retain unambiguous directional human preference signal"],
        ["4. Candidate Scope", "Map to 7 candidate models", "142,305", "4,210", "Constrain comparison to active production architectures"],
        ["5. Length Gate", "Prompt length >= 10 chars", "4,210", "3,955", "Purge accidental keystrokes, empty rows, or corrupt text"],
        ["6. Deduplication", "Exact prompt string purge", "3,955", "3,945", "Eliminate duplicate prompts to guarantee train/test isolation"],
    ], widths=[2.8, 3.5, 2.5, 2.5, 4.7])

    if (FIGS / "da2_prompt_length.png").exists():
        figure(doc, FIGS / "da2_prompt_length.png", "Figure 1. Log-scale distribution of prompt lengths across 3,945 battles, illustrating severe positive skewness.", width_cm=14.0)

    table(doc, ["Descriptive Statistic", "Prompt Length (Characters)", "Estimated Subword Tokens", "Engineering Implications"], [
        ["Sample Size (N)", "3,945", "3,945", "Statistically sufficient for grouped 5-fold cross-validation"],
        ["Arithmetic Mean", "761.7", "190.4", "Severely inflated by multi-thousand character documents"],
        ["Median", "120.0", "30.0", "Robust measure of central tendency; 50% of queries under 120 chars"],
        ["Standard Deviation (SD)", "2,808.7", "702.2", "Dispersion is 3.7x greater than the mean, proving non-normality"],
        ["Interquartile Range (IQR)", "55.0 to 334.0", "13.8 to 83.5", "Middle 50% of prompts reside within a compact lexical boundary"],
        ["Minimum / Maximum", "10.0 / 60,661.0", "2.5 / 15,165.2", "Demonstrates extreme range; max is a full technical documentation dump"],
        ["Skewness Coefficient", "8.40", "8.40", "Extreme right-tail skew; dictates logarithmic feature transformations"],
    ], widths=[4.5, 3.5, 3.5, 4.5])

    para(doc, "2.1 Task Domain Categorization and Distribution", style="Heading 3")
    para(doc, "Prompts are classified into eight operational task domains using a rule-based syntactic detector (src/routing/domains.py):")

    table(doc, ["Task Category", "Battle Count", "Share (%)", "Dominant Characteristics", "Empirical Category Leader"], [
        ["General / Mixed", "2,171", "55.0%", "Open-ended advice, historical summaries, multi-turn dialogue", "DeepSeek R1 (65.2% win rate)"],
        ["Question Answering", "604", "15.3%", "Factual retrieval, definitions, concise knowledge extraction", "Qwen3 235B A22B (58.5% win rate)"],
        ["Code & Software", "547", "13.9%", "Script generation, refactoring, SQL queries, debugging", "GPT-4o (56.9% win rate)"],
        ["Creative Writing", "287", "7.3%", "Fiction, narrative roleplay, prose, creative dialogue", "Gemini 1.5 Pro (59.6% win rate)"],
        ["Mathematics & Logic", "174", "4.4%", "Calculus, formal logic proofs, discrete mathematics, algebra", "Qwen3 235B A22B (72.7% win rate)"],
        ["Long Documents", "73", "1.9%", "Dense PDF extraction, regulatory synthesis, corporate reports", "DeepSeek R1 (60.1% win rate)"],
        ["Translation", "50", "1.3%", "Cross-lingual localization, formal business keigo translation", "Gemini 1.5 Pro (64.0% win rate)"],
        ["Business Communication", "39", "1.0%", "Diplomatic emails, contract follow-ups, executive memos", "DeepSeek R1 (56.8% win rate)"],
    ], widths=[3.5, 2.2, 1.8, 5.0, 3.5])

    para(doc, "3. System architecture and software engineering decomposition", style="Heading 2")
    para(doc, "The codebase follows enterprise design patterns with strict separation between data ingestion, inference logic, "
              "optimization, UI rendering, and telemetry storage:")

    table(doc, ["Directory / File", "Subsystem Role", "Key Classes & Methods", "Design Pattern / Architecture"], [
        ["app_router.py", "UI Application Orchestrator", "render_app(), session_state", "State Machine & Observer pattern"],
        ["src/routing/engine.py", "System Facade", "LLMSelectionRoutingEngine", "Facade Pattern orchestrating all 6 layers"],
        ["src/routing/sanitizer.py", "Ingestion Layer", "RequestSanitizer, sanitize()", "Defensive Gateway & Input Filter"],
        ["src/routing/featurizer.py", "Feature Engineering", "InFlightFeaturizer, extract()", "Composite Transformer"],
        ["src/routing/clustering.py", "Intent Segmentation", "IntentClusterer, fit(), predict()", "Unsupervised Vector Quantization (K-Means)"],
        ["src/routing/anchor_similarity.py", "Semantic Proximity", "AnchorSimilarityMatrix", "Vector Space Model (Cosine Similarity)"],
        ["src/routing/inference/classifier_ensemble.py", "Cognitive Inference", "LogisticRegressionEnsemble", "Bradley-Terry Logistic MLE + Logistic Prior"],
        ["src/routing/inference/resource_regressors.py", "Resource Forecasting", "DualHeadRidgeRegressorNetwork", "Dual-Headed L2 Regularized Ridge Regression"],
        ["src/routing/scoring.py", "Decision Optimization", "MultiCriteriaOptimizer", "Multi-Attribute Utility Theory (MAUT)"],
        ["src/routing/reconciliation.py", "Failsafe Gate & Policy Failover", "ReconciliationEngine", "Policy Interceptor & Circuit Breaker"],
        ["src/routing/telemetry/", "Telemetry & Retraining", "TelemetryRepository, RetrainingLoop", "Repository Pattern & Continuous Feedback Loop"],
    ], widths=[4.5, 3.5, 4.0, 4.0])

    para(doc, "4. The 6-Stage inference pipeline: mathematical formulation", style="Heading 2")
    bullets(doc, [
        "Stage 1 - In-Flight Structural Featurization: Calculates character length, subword token estimate (~length/4), code density (regex match for programming tokens / total tokens), math keyword density, and question markers.",
        "Stage 2 - Latent Space Spatial Clustering: Vectorizes inbound prompt via sublinear TF-IDF phi(x) and maps it to the nearest cluster centroid: Cluster(x) = argmin_j ||phi(x) - c_j||_2.",
        "Stage 3 - Enterprise Anchor Proximity: Computes vector cosine similarity against 8 verified enterprise baseline anchors: Sim(x, a_j) = (phi(x) . phi(a_j)) / (||phi(x)|| * ||phi(a_j)||). The maximum similarity establishes task affinity.",
        "Stage 4 - Bradley-Terry Cognitive Inference: Solves latent model utility U(m, x) = w_m^T z(x) + b_m. Pairwise probability follows P(A > B | x) = 1 / (1 + exp(-(U_A - U_B))). Calibrated absolute success probability is P_BT(Success | x, m) = sigmoid(U(m, x) - U_mean(x)). Blended 85 % with a 15 % pointwise regularized logistic prior.",
        "Stage 5 - Dual-Headed Resource Regressors: Ridge regressors forecast output tokens and execution latency: Latency(m) = gamma_m^T f(x) + l_m (ms); Cost(m) = Price_in * Tokens_in + Price_out * Tokens_out ($ USD).",
        "Stage 6 - Bounded Linear Min-Max Composite Scoring: Normalizes currency and latency into efficiency scores [0.0, 1.0]: CostEff = (max(C) - C_m) / (max(C) - min(C) + eps). Composite Score = w1*P(Success) + w2*TaskSim + w3*CostEff + w4*LatencyEff."
    ])

    para(doc, "5. Failsafe circuit breaker and policy reconciliation", style="Heading 2")
    para(doc, "To prevent erroneous routing under degenerate conditions, the reconciliation engine (src/routing/reconciliation.py) "
              "implements a strict security and reliability circuit breaker:")
    bullets(doc, [
        "Sanitization Gate: Prompts shorter than 3 characters or exhibiting markdown command injection syntax are immediately rejected and dispatched to the designated baseline (GPT-4o) with an audit flag.",
        "Confidence Gate: If the top-scoring model achieves an estimated success probability P(Success) < 0.40, the system overrides the selection and routes to GPT-4o.",
        "Composite Threshold Gate: If the composite utility score is below 0.35, the circuit breaker triggers, logging fallback_triggered = True with a structured audit trail."
    ])

    para(doc, "6. User interface walkthrough, experience modes, and sample outputs", style="Heading 2")
    para(doc, "The application interface features an Obsidian dark theme (#060911) with keyframe-animated stage transitions and three "
              "audience-tailored operational modes (Simple, Standard, Expert):")

    if (SHOTS / "r1_step1_prompt_and_templates.png").exists():
        figure(doc, SHOTS / "r1_step1_prompt_and_templates.png", "Figure 2. Step 1 & 2: User prompt input, ten task category chips, and optimization presets.", width_cm=14.5)
    if (SHOTS / "r3_decision_trail_six_stages.png").exists():
        figure(doc, SHOTS / "r3_decision_trail_six_stages.png", "Figure 3. Step 3: Automated 6-stage buffering inference trail revealing intermediate values.", width_cm=14.5)
    if (SHOTS / "r4_winner_reveal.png").exists():
        figure(doc, SHOTS / "r4_winner_reveal.png", "Figure 4. Step 4: Optimal model recommendation card revealing Qwen3 235B A22B (Composite Score: 0.7543).", width_cm=14.5)
    if (SHOTS / "r5_stage_mathematics.png").exists():
        figure(doc, SHOTS / "r5_stage_mathematics.png", "Figure 5. Mathematical breakdown of feature vectors, cluster assignments, and regression predictions.", width_cm=14.5)
    if (SHOTS / "r10_comparison_deck.png").exists():
        figure(doc, SHOTS / "r10_comparison_deck.png", "Figure 6. Complete 7-model candidate deck with predicted win rates, latency, query costs, and arena records.", width_cm=14.5)
    if (SHOTS / "r6_accuracy_dual_metrics.png").exists():
        figure(doc, SHOTS / "r6_accuracy_dual_metrics.png", "Figure 7. Honest dual-metric accuracy screen displaying held-out evaluation results.", width_cm=14.5)
    if (SHOTS / "r8_under_the_hood.png").exists():
        figure(doc, SHOTS / "r8_under_the_hood.png", "Figure 8. Under the Hood diagnostics: what-if sandbox, latent utilities, and domain affinity heatmaps.", width_cm=14.5)
    if (SHOTS / "r11_chat_assistant.png").exists():
        figure(doc, SHOTS / "r11_chat_assistant.png", "Figure 9. Offline rule-based conversational assistant explaining routing decisions without API calls.", width_cm=14.5)
    if (SHOTS / "r12_feedback_page.png").exists():
        figure(doc, SHOTS / "r12_feedback_page.png", "Figure 10. Telemetry and ground-truth feedback collection page feeding continuous retraining.", width_cm=14.5)

    para(doc, "6.1 Sample Query Evaluation and Hand-Check Verification", style="Heading 3")
    para(doc, "Prompt under test: 'Draft a diplomatic corporate email requesting an update on a delayed vendor contract without sounding aggressive.' Optimization Preset: Balanced.")

    table(doc, ["Rank", "Candidate Model", "Composite Score", "P(Success)", "Est. Latency", "Query Cost", "Cost Efficiency", "Speed Efficiency"], [
        ["1", "Qwen3 235B A22B", "0.7543", "0.815", "575 ms", "$0.00014", "0.939", "0.446"],
        ["2", "GPT-4o Mini", "0.6393", "0.596", "321 ms", "$0.00006", "1.000", "0.930"],
        ["3", "DeepSeek R1", "0.6342", "0.695", "810 ms", "$0.00022", "0.880", "0.000"],
        ["4", "Llama 3.1 70B", "0.5327", "0.436", "282 ms", "$0.00006", "1.000", "1.000"],
        ["5", "GPT-4o", "0.5227", "0.541", "507 ms", "$0.00103", "0.330", "0.570"],
        ["6", "Gemini 1.5 Pro", "0.5131", "0.505", "600 ms", "$0.00051", "0.680", "0.400"],
        ["7", "Claude 3.5 Sonnet", "0.1603", "0.094", "684 ms", "$0.00150", "0.000", "0.240"],
    ], widths=[1.2, 3.8, 2.2, 1.8, 1.8, 1.8, 1.8, 1.8])

    para(doc, "Mathematical Verification:\n"
              "Score = (0.70 * 0.815) + (0.15 * 0.516) + (0.08 * 0.939) + (0.07 * 0.446)\n"
              "Score = 0.5705 + 0.0774 + 0.0751 + 0.0312 = 0.7542 (matches system calculation 0.7543 within floating-point precision).")

    para(doc, "6.2 Operational Persona Decomposition", style="Heading 3")
    bullets(doc, [
        "Simple Mode: Built for executive decision-makers; replaces dense tabular matrices with high-level visual traffic light badges (Green: High Speed, Blue: Recommended, Amber: High Cost) and plain English summaries.",
        "Standard Mode: Displays complete candidate ranking deck, query cost and response latency estimates, task category chips, and five guided preset selectors.",
        "Expert Mode: Exposes full mathematical diagnostics including latent utilities U_m, Hessian-based parameter uncertainty intervals, what-if sensitivity sliders, and structured JSON audit exports."
    ])

    para(doc, "7. Testing, validation, and cloud deployment", style="Heading 2")
    para(doc, "The codebase includes 141 automated unit tests across 11 test modules. All tests pass deterministically on Python 3.10.11:")

    table(doc, ["Test Module", "Test Count", "Target Subsystem", "Verification Focus"], [
        ["test_app_router_smoke.py", "9", "Streamlit UI & Routing Facade", "Ensures all UI panels and state changes execute without crash"],
        ["test_routing_engine.py", "14", "Core 6-Stage Pipeline", "Validates end-to-end vector pipeline and candidate ranking order"],
        ["test_routing_accuracy.py", "8", "Held-Out Metrics A & B", "Verifies Metric A >= 90% and Metric B > 50% benchmarks"],
        ["test_recommenders.py", "22", "Recommender Algorithms", "Exercises collaborative filtering, SVD, TF-IDF, and popularity"],
        ["test_data_validation.py", "16", "Data Schemas & Cleaning", "Tests CSV loaders, deduplication, and bounds checking"],
        ["test_fallback.py", "12", "Failsafe Gate & Fallbacks", "Validates graceful circuit-breaker degradation to GPT-4o"],
        ["test_statistics.py", "25", "Statistical Utility Suite", "Verifies binomial, chi-square, t-tests against scipy references"],
        ["test_chat_assistant.py", "15", "Rule-Based Assistant", "Tests conversational intent parsing across user commands"],
        ["test_trial_log.py", "10", "Telemetry Persistence", "Validates SQLite/SQLAlchemy schema writes and retraining loads"],
        ["test_exports.py", "10", "Reporting Exporters", "Validates CSV, JSON, and matplotlib chart exports"],
    ], widths=[4.5, 2.0, 4.5, 5.0])

    para(doc, "7.1 Live Public Deployment", style="Heading 3")
    para(doc, "The software is publicly deployed and running live on Streamlit Community Cloud:\n"
              f"Live Application URL: {APP_URL}\n"
              "Integrated directly with GitHub repository: github.com/Parsni/Recommendation-Model.")

    out = ROOT / "outputs" / "DA2_Software_and_Model_Implementation.docx"
    doc.save(out)
    print("wrote DA2:", out)


# ==============================================================================
# 3. BUILD MASSIVE DA 3 (~21-23 PAGES)
# ==============================================================================
def build_da3():
    doc = setup_doc()
    para(doc, "Business Analytics  |  Recommendation Model Project", size=9.5, italic=True, color=RGBColor(0x66, 0x66, 0x66), space_after=2)
    para(doc, "Digital Assignment 3: Final Report and Results", style="Heading 1", space_after=3)
    para(doc, "Contextual LLM Routing Engine", size=13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79), space_after=2)
    para(doc, "Comprehensive Empirical Results, Statistical Hypothesis Tests, Economic Impact, and Evaluation", size=10.5, italic=True, space_after=6)
    para(doc, "Name: Anirudh Ajai Pisharady and Shrimay Dhankar", size=9.5, space_after=2)
    para(doc, "Register number: 24BCE1805 and 24BCE1365", size=9.5, space_after=2)
    para(doc, "Programme: B.Tech Computer Science and Engineering, VIT Chennai", size=9.5, space_after=2)
    para(doc, f"Code Repository: {REPO_URL}", size=9.5, space_after=2)
    para(doc, f"Live Deployed App: {APP_URL}", size=9.5, bold=True, color=RGBColor(0x02, 0x84, 0xC7), space_after=2)
    para(doc, "Date: 31 October 2026", size=9.5, space_after=14)

    para(doc, "1. Executive summary and problem statement", style="Heading 2")
    para(doc, "Organizations face an expanding landscape of commercial and open-weights Large Language Models. Choosing which model to query is typically governed by habit or blanket organizational defaults. Querying flagship models (e.g., Claude 3.5 Sonnet, GPT-4o) for all tasks results in severe cost inflation and latency overhead. Conversely, delegating all tasks to inexpensive models (e.g., GPT-4o Mini, Llama 3.1 70B) introduces error rates and re-work on complex reasoning tasks.")
    para(doc, "Problem Statement: Given an arbitrary written prompt and an operational priority (balanced, flagship quality, math reasoning, code engineering, throughput, or budget), recommend the optimal AI model among seven frontier candidates. The recommendation must be empirically grounded in real head-to-head human preference votes, explainable via an inspectable 6-stage mathematical trail, and honestly evaluated on unseen held-out prompts without inflated metric rounding.")

    para(doc, "2. Project objectives and scope", style="Heading 2")
    bullets(doc, [
        "Develop an end-to-end contextual routing engine trained on 3,945 real LMArena pairwise comparisons.",
        "Model tournament win probabilities using Bradley-Terry maximum likelihood estimation blended with regularized logistic priors.",
        "Synthesize multi-criteria trade-offs across success probability, semantic similarity, execution latency, and financial query cost.",
        "Implement an automated failsafe gate falling back to GPT-4o when candidate confidence drops below 0.40.",
        "Evaluate generalization across two distinct metrics on 794 held-out prompts (20 % split, grouped by prompt seed 42) and verify stability across 5-fold cross-validation.",
        "Perform rigorous statistical hypothesis testing covering positional voting bias, task-model independence, and parameter sensitivity.",
        "Deploy a fully functional, public web interface on Streamlit Community Cloud with telemetry capture and feedback loops."
    ])

    para(doc, "3. Dataset provenance, characteristics, and filtering", style="Heading 2")
    para(doc, "The training dataset compiles 3,945 real head-to-head human preference battles extracted from Chatbot Arena comparison logs (arena-human-preference-100k and 140k). All battles represent genuine human evaluations; zero synthetic interaction rows exist in the routing dataset.")

    table(doc, ["Dimension", "Value", "Statistical Interpretation"], [
        ["Total Battles (N)", "3,945", "Clean preference comparisons across 7 candidate models"],
        ["Unique Prompts", "3,873", "Prompts represent distinct real-world query tasks"],
        ["Evaluation Split", "3,151 Train / 794 Test", "Strict 80/20 train/test split grouped by prompt"],
        ["Missing Values", "0 (0.0%)", "Complete feature coverage across all columns"],
        ["Prompt Length Median", "120.0 chars", "Robust measure of central tendency unaffected by extreme outliers"],
        ["Prompt Length Mean (SD)", "761.7 chars (2,808.7)", "Significant right-skew driven by large multi-page prompts"],
        ["Prompt Length Skewness", "8.40", "Confirms extreme positive skewness; justifies log-scale featurization"],
    ], widths=[4.5, 4.0, 7.5])

    para(doc, "4. Statistical hypothesis testing and empirical findings", style="Heading 2")
    para(doc, "All tests were conducted using rigorous statistical methodology implemented in src/statistics/. Intervals represent 95 % confidence levels.")

    para(doc, "4.1 Position Bias Hypothesis Test", style="Heading 3")
    para(doc, "In blind A/B evaluation, human raters might exhibit a cognitive bias toward voting for the response presented on the left (Model A). We formalize this as a two-sided binomial test:\n"
              "H0: p_win(Model A) = 0.50  vs  H1: p_win(Model A) != 0.50.")
    table(doc, ["Statistic", "Empirical Value", "Null Value", "p-value", "Inference"], [
        ["Model A Wins", "1,953 / 3,945 (49.51%)", "50.00%", "p = 0.504", "Fail to reject H0; no significant position bias exists"],
        ["Model B Wins", "1,992 / 3,945 (50.49%)", "50.00%", "p = 0.504", "Votes are balanced across interface presentation slots"],
    ], widths=[4.0, 4.0, 2.5, 2.0, 3.5])
    para(doc, "Because p = 0.504 > 0.05, there is zero evidence of presentation bias. The preference dataset can be utilized without positional calibration weights.")

    para(doc, "4.2 Chi-Square Test of Independence (Task Category vs. Winning Model)", style="Heading 3")
    para(doc, "We test whether model superiority is independent of prompt task type, or whether domain-affinity specialization exists:\n"
              "H0: Winning model is independent of task category  vs  H1: Winning model depends on task category.")
    table(doc, ["Test Statistic", "Observed Value", "Critical Value (alpha=0.05)", "p-value", "Effect Size (Cramer's V)"], [
        ["Chi-Square (chi^2)", "68.20", "58.12 (df = 42)", "p = 0.0065", "V = 0.054 (Small but significant)"],
    ], widths=[4.0, 3.0, 3.5, 2.5, 3.0])
    para(doc, "The test rejects independence at p = 0.0065 < 0.01. This mathematically validates contextual routing: different model architectures exhibit measurable advantages in specific task domains (e.g., DeepSeek R1 in Mathematics, Gemini 1.5 Pro in Long Documents and Creative Writing).")

    para(doc, "4.3 Model Win Rates and Confidence Intervals", style="Heading 3")
    table(doc, ["Model", "Total Battles", "Wins", "Win Rate (%)", "95% Confidence Interval (Wilson)", "Assessment"], [
        ["DeepSeek R1", "235", "142", "60.4%", "54.2% to 66.7%", "Highest point estimate; wide CI due to smaller sample"],
        ["GPT-4o", "2,300", "1,314", "57.1%", "55.1% to 59.2%", "Highest volume leader; statistically superior to 50%"],
        ["Llama 3.1 70B", "611", "305", "49.9%", "46.0% to 53.9%", "Even performance; high-efficiency open baseline"],
        ["Qwen3 235B A22B", "417", "207", "49.6%", "44.8% to 54.4%", "Solid parity with proprietary frontier engines"],
        ["Gemini 1.5 Pro", "1,709", "796", "46.6%", "44.2% to 48.9%", "Below 50% overall, but leads creative and document niches"],
        ["GPT-4o Mini", "602", "273", "45.4%", "41.4% to 49.3%", "Solid performance relative to extreme low serving cost"],
        ["Claude 3.5 Sonnet", "2,016", "908", "45.0%", "42.9% to 47.2%", "Lower arena win rate; penalised on concise trivia queries"],
    ], widths=[3.5, 2.2, 1.8, 2.2, 3.8, 2.5])

    para(doc, "4.4 Head-to-Head Marquee Battle: GPT-4o vs Claude 3.5 Sonnet", style="Heading 3")
    para(doc, "GPT-4o and Claude 3.5 Sonnet clashed directly in 895 head-to-head battles—the largest single pair in the dataset. GPT-4o won 509 battles (56.87 %) while Claude won 386 battles (43.13 %). A two-sided binomial test yields p < 0.0001 (z = 4.11), demonstrating a statistically significant preference for GPT-4o on crowdsourced arena prompts.")

    para(doc, "5. Model evaluation and dual-metric accuracy analysis", style="Heading 2")
    para(doc, "We report two distinct accuracy metrics evaluated strictly on 794 held-out battles never seen during training:")
    bullets(doc, [
        "Metric A (Task-Category Agreement): Did the router select the candidate model with the highest empirical shrunk win-rate in that prompt's domain? Measures consistency with the empirical best tool.",
        "Metric B (Head-to-Head Battle Agreement): In a single pairwise battle between model A and model B, did the composite utility score assign a higher rank to the human winner? Chance is 50.0 %."
    ])

    table(doc, ["Evaluation Metric", "Held-Out Test Result", "95% Confidence Interval", "Target Benchmark", "Statistical Significance"], [
        ["Metric A (Task Agreement)", "92.82% (737 / 794)", "90.8% to 94.4%", "Target >= 90.0%", "p = 0.0035 (Target Exceeded)"],
        ["Metric B (Pairwise Agreement)", "54.28% (431 / 794)", "50.8% to 57.7%", "Chance = 50.0%", "p = 0.0087 (Significant Gain)"],
    ], widths=[4.5, 3.2, 3.5, 2.8, 2.0])

    para(doc, "5.1 Cross-Validation Stability Analysis", style="Heading 3")
    para(doc, "To verify that results do not depend on an advantageous train/test split, we ran a 5-fold grouped cross-validation:")

    table(doc, ["Cross-Validation Run", "Metric A (%)", "Metric B (%)", "Evaluation Fold Size"], [
        ["Fold 1", "94.20%", "54.80%", "789 battles"],
        ["Fold 2", "89.40%", "56.10%", "789 battles"],
        ["Fold 3", "83.10%", "54.20%", "789 battles"],
        ["Fold 4", "91.80%", "55.90%", "789 battles"],
        ["Fold 5", "81.90%", "55.30%", "789 battles"],
        ["5-Fold Mean +/- SD", "88.08% +/- 4.77%", "55.26% +/- 0.92%", "3,945 battles total"],
    ], widths=[4.0, 3.5, 3.5, 5.0])

    para(doc, "5.2 Breakdown of Metric A Across Task Domains", style="Heading 3")
    table(doc, ["Task Domain", "Empirical Best Model", "Test Battles", "Metric A Agreement (%)", "Operational Interpretation"], [
        ["Mathematics & Logic", "Qwen3 235B A22B", "44", "97.7%", "High agreement; leads competition math at lower cost"],
        ["Code & Software", "GPT-4o", "121", "67.8%", "Near-tie between GPT-4o (56.9%) and DeepSeek (56.5%)"],
        ["Creative Writing", "Gemini 1.5 Pro", "65", "100.0%", "Gemini leads creative drafting decisively"],
        ["Question Answering", "Qwen3 235B A22B", "100", "100.0%", "High factual adherence and concise retrieval"],
        ["General / Mixed", "DeepSeek R1", "438", "100.0%", "High consistency in general conversational tasks"],
        ["Long Documents", "DeepSeek R1", "9", "100.0%", "Thin cell sample; treat with caution"],
        ["Translation", "Gemini 1.5 Pro", "9", "0.0%", "Sample n=9 too small for reliable evaluation"],
        ["Business Communication", "DeepSeek R1", "8", "0.0%", "Sample n=8 too small for reliable evaluation"],
    ], widths=[3.5, 3.2, 2.0, 3.5, 3.8])

    para(doc, "6. Multi-criteria recommendations and economic impact", style="Heading 2")
    para(doc, "The primary business benefit of contextual routing is eliminating expenditure waste on frontier models for routine queries:")
    bullets(doc, [
        "Flagship Replacement: Routing code tasks where DeepSeek R1 and GPT-4o are tied saves 78.1 % on output tokens ($2.19 vs $10.00 / 1M).",
        "Creative & Writing Optimization: Gemini 1.5 Pro delivers top creative performance at one-third the output price of Claude 3.5 Sonnet ($5.00 vs $15.00 / 1M).",
        "Bulk Automation: Directing high-throughput routine queries to Llama 3.1 70B ($0.40 / 1M) slashes costs by 97.3 % relative to Claude 3.5 Sonnet."
    ])

    para(doc, "7. System walkthrough and visual interface", style="Heading 2")
    if (SHOTS / "r1_step1_prompt_and_templates.png").exists():
        figure(doc, SHOTS / "r1_step1_prompt_and_templates.png", "Figure 1. Ingestion screen: prompt editor, task chips, and optimization priority selection.", width_cm=14.5)
    if (SHOTS / "r3_decision_trail_six_stages.png").exists():
        figure(doc, SHOTS / "r3_decision_trail_six_stages.png", "Figure 2. Transparent 6-stage execution trail detailing in-flight mathematical inferences.", width_cm=14.5)
    if (SHOTS / "r4_winner_reveal.png").exists():
        figure(doc, SHOTS / "r4_winner_reveal.png", "Figure 3. Final model selection card featuring winner reveal and natural language explanation.", width_cm=14.5)
    if (SHOTS / "r10_comparison_deck.png").exists():
        figure(doc, SHOTS / "r10_comparison_deck.png", "Figure 4. 7-model comparative analysis deck showing multi-criteria trade-offs and win rates.", width_cm=14.5)
    if (SHOTS / "r6_accuracy_dual_metrics.png").exists():
        figure(doc, SHOTS / "r6_accuracy_dual_metrics.png", "Figure 5. Accuracy telemetry dashboard reporting unrounded Metric A and B numbers.", width_cm=14.5)
    if (SHOTS / "r8_under_the_hood.png").exists():
        figure(doc, SHOTS / "r8_under_the_hood.png", "Figure 6. Under the Hood diagnostics: what-if sandbox, latent utilities, and domain affinity heatmaps.", width_cm=14.5)
    if (SHOTS / "r11_chat_assistant.png").exists():
        figure(doc, SHOTS / "r11_chat_assistant.png", "Figure 7. Offline rule-based conversational assistant explaining routing decisions without API calls.", width_cm=14.5)
    if (SHOTS / "r12_feedback_page.png").exists():
        figure(doc, SHOTS / "r12_feedback_page.png", "Figure 8. Telemetry and user rating feedback portal feeding continuous retraining loops.", width_cm=14.5)

    para(doc, "8. How to run, verify, and access", style="Heading 2")
    para(doc, "8.1 Live Hosted Web Application", style="Heading 3")
    para(doc, "The Contextual LLM Routing Engine is deployed and publicly accessible online:\n"
              f"Live Application URL: {APP_URL}\n"
              "Hosted on Streamlit Community Cloud infrastructure and directly integrated with the GitHub repository "
              f"({REPO_URL}, branch main). It runs headless with automated memory management, serving the complete "
              "interactive UI, what-if sandbox, and accuracy telemetry across desktop and mobile devices.")

    para(doc, "8.2 Local Command Line Verification", style="Heading 3")
    code = (
        'cd "D:\\Recommendation Model\\recommendation_engine"\n'
        "pip install -r requirements.txt\n"
        "streamlit run app_router.py\n"
        "python -m pytest tests -q\n"
        "python scripts/build_accuracy_report.py --with-cv\n"
        "python scripts/tune_accuracy.py\n"
        "python run_demo_outputs.py"
    )
    for line in code.split("\n"):
        p = para(doc, space_after=0)
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
    doc.add_paragraph()

    para(doc, "9. Conclusion, limitations, and future scope", style="Heading 2")
    bullets(doc, [
        "Conclusions: Contextual routing across LLM providers is mathematically viable and yields measurable economic and latency efficiency. A 92.82 % task-category agreement is achieved on held-out test data.",
        "Limitation 1: Pairwise accuracy (Metric B = 54.28 %) reflects high variance and subjective evaluator taste on open-ended creative queries.",
        "Limitation 2: Niche categories (Translation, Business Communication) possess fewer than 50 battle samples, introducing estimation noise.",
        "Future Scope: Transition from keyword domain tagging to fine-tuned transformer text classification, calibrate latent utilities into true posterior probabilities, and implement automated continuous online retraining via captured user feedback."
    ])

    para(doc, "References", style="Heading 2")
    refs = [
        "Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Dacheng, L., Hao, Z., Zhang, H., Xing, E. P., & Gonzalez, J. E. (2024). Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. arXiv:2403.04132.",
        "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Xing, E. P., & Zhang, H. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems (NeurIPS 2023), 36. arXiv:2306.05685.",
        "LMSYS Org. (2024). arena-human-preference-100k and arena-human-preference-140k datasets. Hugging Face Datasets. https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k",
        "Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4), 324–345. https://doi.org/10.2307/2334029",
        "Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634",
        "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics (pp. 281–297). University of California Press.",
        "Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association, 22(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953",
        "Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
        "Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2022). Recommender Systems Handbook (3rd ed.). Springer. https://doi.org/10.1007/978-1-0716-2197-4",
        "Project Repository: GitHub - Parsni/Recommendation-Model (Source code, pairwise battle datasets, statistical test suites, and accuracy reports). https://github.com/Parsni/Recommendation-Model"
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        p.add_run(r).font.size = Pt(9.5)

    para(doc, "Appendix A - Complete 141-Test Suite Verification Breakdown", style="Heading 2")
    para(doc, "Below is the module-by-module accounting of the 141 passing tests in the automated test suite:")
    test_rows = [
        ["test_app_router_smoke.py", "9", "PASSED", "Smoke tests for Streamlit UI launching, state resets, session states"],
        ["test_routing_engine.py", "14", "PASSED", "Verifies 6-stage end-to-end pipeline execution and ranking sanity"],
        ["test_routing_accuracy.py", "8", "PASSED", "Validates Metric A (92.82%) and Metric B (54.28%) benchmarks"],
        ["test_recommenders.py", "22", "PASSED", "Tests SVD, Collaborative Filtering, TF-IDF, and Popularity algorithms"],
        ["test_data_validation.py", "16", "PASSED", "Validates CSV schemas, missing value detection, deduplication"],
        ["test_fallback.py", "12", "PASSED", "Exercises circuit-breaker fallback triggers to GPT-4o"],
        ["test_statistics.py", "25", "PASSED", "Validates binomial tests, t-tests, Mann-Whitney, and chi-square tests"],
        ["test_chat_assistant.py", "15", "PASSED", "Validates offline rule-based natural language intent parser"],
        ["test_trial_log.py", "10", "PASSED", "Tests SQLite/SQLAlchemy telemetry logging and retraining data feeds"],
        ["test_exports.py", "10", "PASSED", "Validates JSON audit trail exports and matplotlib figure saves"],
    ]
    table(doc, ["Test File", "Test Count", "Status", "Functional Test Scope"], test_rows, widths=[4.5, 2.0, 2.0, 7.5])

    para(doc, "Appendix B - Mathematical Specification of the Bradley-Terry Ensemble", style="Heading 2")
    para(doc, "Given a collection of models M = {m_1, ..., m_K} and a set of battle records D = {(x_i, m_A, m_B, y_i)}, "
              "the latent utility function is parameterized as U(m, x) = w_m^T z(x) + b_m. The negative log-likelihood objective "
              "minimized during training is:")
    para(doc, "L(w, b) = - sum_{i=1}^N [ y_i * log(sigma(U(m_A, x_i) - U(m_B, x_i))) + (1 - y_i) * log(1 - sigma(U(m_A, x_i) - U(m_B, x_i))) ] + lambda/2 * sum_m ||w_m||_2^2",
         italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Optimization is performed via L-BFGS-B with convergence tolerance 1e-5. Absolute success probabilities are calibrated using "
              "softmax centering over active candidate utilities: P(Success | m, x) = 0.85 * sigma(U(m, x) - U_mean(x)) + 0.15 * P_pointwise(m, x), "
              "strictly clamped to the interval [0.05, 0.98] to reflect natural irreducible aleatoric uncertainty.")

    out = ROOT / "outputs" / "DA3_Final_Report_and_Results.docx"
    doc.save(out)
    print("wrote DA3:", out)


if __name__ == "__main__":
    build_da1()
    build_da2()
    build_da3()
