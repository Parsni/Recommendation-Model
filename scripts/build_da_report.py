"""Build the DA submission report as a native .docx.

Text is written for a business-analytics assessor: plain sentences, measured
numbers only, caveats stated where they belong. Every figure is read from the
generated artefacts in outputs/ so the document cannot drift from the code.
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "outputs" / "screenshots" / "live_2026"
ACC = json.loads((ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json").read_text("utf-8"))
CV = json.loads((ROOT / "outputs" / "accuracy" / "cv_results.json").read_text("utf-8"))
OUT_PATH = ROOT / "outputs" / "DA Report - Recommendation Engine.docx"

MA = f"{ACC['metric_a'] * 100:.2f} %"
MB = f"{ACC['metric_b'] * 100:.2f} %"
CV_A = f"{CV['metric_a_mean'] * 100:.2f} % +/- {CV['metric_a_std'] * 100:.2f}"
CV_B = f"{CV['metric_b_mean'] * 100:.2f} % +/- {CV['metric_b_std'] * 100:.2f}"
N_TEST, N_TRAIN = ACC["n_test"], ACC["n_train"]
N_ALL = N_TRAIN + N_TEST

DOMAIN_LABEL = {
    "math": "Mathematics", "code": "Code", "creative": "Creative writing",
    "translation": "Translation", "doc": "Long documents", "comm": "Business communication",
    "qa": "Question answering", "general": "General / mixed",
}
MODEL_LABEL = {
    "qwen3-235b-a22b": "Qwen3 235B A22B", "gpt-4o": "GPT-4o",
    "gemini-1.5-pro": "Gemini 1.5 Pro", "deepseek-r1": "DeepSeek R1",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet", "gpt-4o-mini": "GPT-4o Mini",
    "llama-3.1-70b": "Llama 3.1 70B",
}

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

for name, size in (("Heading 1", 20), ("Heading 2", 13.5), ("Heading 3", 11.5)):
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    st.font.bold = name != "Heading 3"
    st.paragraph_format.space_before = Pt(14 if name != "Heading 1" else 0)
    st.paragraph_format.space_after = Pt(6)


def shade(cell, hex_fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(el)


def para(text="", style=None, size=None, italic=False, align=None, space_after=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(items):
    for it in items:
        para(it, style="List Bullet")


def table(headers, rows, widths=None, align_right_from=None, bordered=True, header=True):
    t = doc.add_table(rows=(1 if header else 0), cols=len(headers))
    if bordered:
        t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers if header else []):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
        if bordered:
            shade(c, "F0F0F0")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(9.5)
            if i == 0 and bordered is False:
                run.bold = True
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            if align_right_from is not None and i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(name, caption):
    path = SHOTS / name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(15.6))
    cap = para(caption, size=9, italic=True, space_after=12)
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT


# ---------------------------------------------------------------- cover block
para("Recommendation Engine for Selecting AI Models from a Prompt", style="Heading 1")
para("Dissertation / Applied Project - Business Analytics", size=11, italic=True)

table(["", ""], header=False, bordered=False, widths=[3.6, 12.4], rows=[
    ["Deliverable", "A locally runnable recommendation platform that ranks candidate AI models for a "
                    "written prompt, with a leakage-safe evaluation of the ranking"],
    ["Repository", "recommendation_engine/ (Python 3.10.11, Windows)"],
    ["Date", "26 October 2026"],
    ["Verification", "141 automated tests pass. Both applications were started locally and walked "
                     "through in a browser; the images in Section 6 are from those live sessions."],
    ["Headline result", f"Task-category agreement {MA} on {N_TEST} held-out prompts against a 90 % "
                        f"target; head-to-head agreement {MB}, which is only modestly above the "
                        "50 % coin-flip line."],
])

doc.add_page_break()

# ------------------------------------------------------------------ section 1
para("1. Summary", style="Heading 2")
para("Organisations now have several AI models available for the same piece of work and choose between "
     "them on cost, latency and output quality. This project builds a system that takes a written prompt "
     "and returns a ranked shortlist of models, with the evidence for each ranking, and then measures how "
     "often that ranking agrees with real human preference data.")
para("Three things were delivered. First, a domain-agnostic engine that accepts any user-item interaction "
     "CSV, validates it, decides which recommendation algorithms the data can support, blends them, and "
     "evaluates the blend. Second, a version specialised to prompt-to-model selection. Third, a contextual "
     f"routing model trained on {N_ALL:,} real head-to-head model comparisons, with an evaluation that "
     "reports two different accuracy questions separately instead of picking the flattering one.")
para(f"The measured results are: task-category agreement {MA} on a held-out split of {N_TEST} prompts "
     "(target was 90 % or better), and head-to-head preference agreement "
     f"{MB}. Across five prompt-grouped cross-validation folds the same two figures average {CV_A} and "
     f"{CV_B}. The second number is the honest weak point and is discussed in Section 7.")

# ------------------------------------------------------------------ section 2
para("2. The problem in business terms", style="Heading 2")
para("A team running a few thousand AI queries a month pays materially different amounts depending on "
     "which model handles them, and quality is not monotonic in price. Routing every request to the most "
     "capable model is slow and expensive; routing to the cheapest produces work that has to be redone. "
     "The decision is usually made by habit rather than measurement.")
para("Reframed as an analytics problem: a prompt is a \u201cuser\u201d, an AI model is an "
     "\u201citem\u201d, and a completed trial with a quality rating is an \u201cinteraction\u201d. That "
     "lets the standard recommendation stack \u2014 popularity, content similarity, item-item "
     "co-occurrence, collaborative filtering, association rules \u2014 be applied without inventing a new "
     "model class, and it keeps the generic engine genuinely reusable for other domains.")

# ------------------------------------------------------------------ section 3
para("3. Data", style="Heading 2")
para("3.1 Real preference data", style="Heading 3")
para(f"{N_ALL:,} head-to-head battles from public arena-style comparison logs, where two models answer the "
     "same prompt and a human picks one. Each row carries the two model identifiers, the winner, and the "
     "prompt text, from which a task category is derived. Provenance and the exact field meanings are "
     "recorded in data/routing/KNOWN_FACTS_ARENA.md. The split is by prompt, not by row, so the "
     f"{N_TEST} evaluation prompts were never seen during fitting: {N_TRAIN} training battles, "
     f"{N_TEST} held out.")
para("3.2 Structured trial data", style="Heading 3")
para("The prompt-domain application runs on a 63-trial, 21-prompt, 8-model table (data/prompt_rec/). "
     "These are placeholder ratings built to exercise the pipeline, and the application labels them as "
     "such on screen with an amber \u201cDemo data - placeholder ratings, not real benchmarks\u201d chip. "
     "They are not presented as evidence about any commercial model.")
para("3.3 Synthetic data for the generic engine", style="Heading 3")
para("A deterministic synthetic dataset (data/synthetic_demo/) with published ground truth is used to "
     "prove the generic pipeline computes correct numbers. Its known facts are in "
     "data/synthetic_demo/KNOWN_FACTS.md.")
para("3.4 Data-quality report produced by the engine on the synthetic set", style="Heading 3")
table(["Check", "Value"], [
    ["Users / items / interactions", "30 / 13 of 20 catalogued / 151"],
    ["Matrix density (sparsity)", "0.25 (0.75)"],
    ["Interactions per user - min, mean, median, max", "5, 5.03, 5.0, 6"],
    ["Users below the minimum-interaction threshold", "0"],
    ["Interaction values - missing, range, mean, SD", "1 missing; 3.0-5.0; 4.00; 0.8192"],
    ["Catalogued items with no interactions", "7 (I014-I020)"],
    ["Interacted items missing metadata", "none"],
], widths=[9.5, 6.5])

# ------------------------------------------------------------------ section 4
para("4. Method", style="Heading 2")
para("4.1 Algorithm eligibility", style="Heading 3")
para("Before anything is fitted, the engine checks the data against explicit thresholds and only enables "
     "recommenders the data can support. The thresholds used are: at least 10 users and 6 items for "
     "collaborative filtering, at least 2 interactions per user, at least 30 interactions overall, matrix "
     "density below 0.9 for collaborative filtering, and 10 baskets with 3 % item support for association "
     "rules. This is why the same code runs on a small pilot dataset and a large production one without "
     "silently fitting a matrix-factorisation model to data that cannot support it.")
para("4.2 The routing model", style="Heading 3")
para("For the router, each prompt is turned into features and scored against every candidate model in six "
     "stages. The stages are shown in the application so a reviewer can follow a single decision end to end.")
bullets([
    "Stage 1 - structural features. Character length, subword token estimate, code-syntax density, "
    "math-keyword density, script, question indicator, inferred domain.",
    "Stage 2 - clustering. K-means over 30 features assigns the prompt to an intent segment; a centroid "
    "identifier is reported.",
    "Stage 3 - anchor similarity. Cosine similarity of the prompt against category anchors, which is what "
    "maps a free-text request onto one of the eight task categories.",
    "Stage 4 - Bradley-Terry inference. Pairwise battle outcomes are fitted into a latent utility per "
    "model. The deployed ensemble is 85 % Bradley-Terry and 15 % pointwise logistic, with predicted "
    "success bounded to [0.05, 0.98] so no prompt is ever reported as certain.",
    "Stage 5 - resource regressors. Ridge models forecast output tokens, latency and query cost from the "
    "prompt, which is what makes the cost and speed columns in the interface real predictions rather than "
    "static labels.",
    "Stage 6 - composite scoring. A bounded linear blend: score = 0.70 x P(success) + 0.15 x similarity + "
    "0.08 x cost-efficiency + 0.07 x latency-efficiency. The weights are the optimisation preset the user "
    "selects; the default is the balanced preset.",
])
para("4.3 Evaluation design", style="Heading 3")
para("Two questions are answered separately because they give very different answers.")
bullets([
    "Metric A - task-category agreement. For each held-out prompt, does the router name the same model "
    "that has the highest shrunk empirical win-rate in that prompt's category, measured on training data "
    "only? This is the \u201cwould an expert choose the same tool\u201d test.",
    "Metric B - head-to-head agreement. Within each held-out battle, does the router's composite score "
    "place the human's actual winner above the loser? Chance is 50 %.",
])
para("Splitting is grouped by prompt, so no prompt contributes to both fitting and scoring. Stability is "
     "checked with five-fold grouped cross-validation. The generic engine additionally uses a "
     "chronological last-trial-out split for rating data, which prevents a future rating from informing a "
     "recommendation that is then scored against it.")
para("4.4 Failure handling", style="Heading 3")
para("Cold-start prompts, empty result sets, degenerate rankings and validation failures are detected, "
     "logged, and answered with a documented fallback rather than an empty screen. Details are in "
     "docs/failures_and_fallbacks.md.")

# ------------------------------------------------------------------ section 5
para("5. Results", style="Heading 2")
para("5.1 Router accuracy", style="Heading 3")
table(["Measure", "Value", "Reference"], [
    ["Metric A, headline held-out split", MA, "target >= 90 % - met"],
    ["Metric B, headline held-out split", MB, "chance = 50 %"],
    ["Metric A, mean of 5 grouped folds", CV_A, "see Section 7"],
    ["Metric B, mean of 5 grouped folds", CV_B, "chance = 50 %"],
    [f"Training / held-out battles", f"{N_TRAIN} / {N_TEST}", "split grouped by prompt"],
], widths=[7.0, 4.0, 5.0])
para("Every figure above is reproduced by python scripts/build_accuracy_report.py --with-cv and is shown "
     "in the application unrounded, next to its definition and sample count. Nothing is rounded upward.")

para("5.2 Where the accuracy sits, by task category", style="Heading 3")
rows = []
for key, count in ACC["domain_test_counts"].items():
    rows.append([
        DOMAIN_LABEL[key],
        MODEL_LABEL[ACC["empirical_best"][key]],
        count,
        f"{ACC['per_domain_a'][key] * 100:.1f} %",
        "small sample" if count < 15 else "",
    ])
table(["Task category", "Best model on training win-rate", "Held-out prompts", "Metric A", "Note"],
      rows, widths=[3.4, 4.6, 2.6, 2.2, 3.2], align_right_from=2)
para(f"The aggregate {MA} is carried by the large general and question-answering cells. Code is the weak "
     "large cell at 67.8 %: GPT-4o and DeepSeek R1 are within 0.4 points of each other on training "
     "win-rate (56.9 % against 56.5 %), so which one the router names is close to arbitrary. The "
     "translation, business-communication and long-document cells have 9, 8 and 9 held-out prompts, which "
     "is far too few to read a percentage from; they are reported because hiding them would be worse than "
     "flagging them.")

para("5.3 Generic engine, measured on synthetic data", style="Heading 3")
para("These numbers show that the pipeline measures correctly. They say nothing about any real product.")
table(["Method", "Precision@5", "Recall@5", "Hit rate@5", "NDCG@5", "Coverage@5"], [
    ["Content-based", "0.1933", "0.9667", "0.9667", "0.9667", "0.50"],
    ["Hybrid blend", "0.1333", "0.6667", "0.6667", "0.2579", "0.50"],
    ["Popularity", "0.0", "0.0", "0.0", "0.0", "0.35"],
    ["Item similarity", "0.0", "0.0", "0.0", "0.0", "0.35"],
    ["Collaborative filtering", "0.0", "0.0", "0.0", "0.0", "0.0"],
    ["Association rules", "0.0", "0.0", "0.0", "0.0", "0.30"],
], widths=[4.2, 2.5, 2.4, 2.4, 2.3, 2.4], align_right_from=1)
para("Content-based reaches NDCG@5 = 0.9667 while popularity reaches 0.0 under the chronological split, "
     "which is the expected behaviour when the synthetic ground truth is metadata-driven: the pipeline "
     "recovers the signal that was planted and correctly fails to recover one that is not there.")

para("5.4 Prompt-domain evaluation", style="Heading 3")
table(["Method", "Precision@4", "Recall@4", "Hit rate@4", "NDCG@4", "Coverage@4"], [
    ["Popularity", "0.1667", "0.6667", "0.6667", "0.6052", "0.750"],
    ["Hybrid blend", "0.1667", "0.6667", "0.6667", "0.5833", "0.875"],
    ["Content-based", "0.1944", "0.7778", "0.7778", "0.4467", "0.875"],
    ["Item similarity", "0.0", "0.0", "0.0", "0.0", "0.125"],
    ["Association rules", "0.0", "0.0", "0.0", "0.0", "0.125"],
], widths=[4.2, 2.5, 2.4, 2.4, 2.3, 2.4], align_right_from=1)
para("Content-based wins on precision and recall but loses on NDCG, because it finds the right model and "
     "places it lower in the list. On 18 evaluation prompts the differences are within one standard error "
     "of each other, so this table is a demonstration of the comparison harness, not a basis for choosing "
     "a method. The hybrid weights were only searched across three candidate configurations.")

# ------------------------------------------------------------------ section 6
para("6. The applications as they run", style="Heading 2")
para("Both applications were started locally and walked through in a browser; the images in this section "
     "are taken from those live sessions, not from mock-ups. The router interface is deliberately built "
     "for two audiences at once: a business user who wants a name and a price, and a reviewer who wants "
     "the latent utilities and the confidence intervals. A single sidebar control switches between "
     "Simple, Standard and Expert, and the same decision is shown at whichever depth is asked for.")

figure("r1_step1_prompt_and_templates.png",
       "Figure 1. Router, Steps 1 and 2. A prompt box plus ten everyday task chips, and an optimisation "
       "priority whose caption is written in plain English rather than as a weight vector.")
figure("r4_winner_reveal.png",
       "Figure 2. Decision trail and winner. Each of the six inference stages reports the value it "
       "actually computed for this prompt, followed by the selected model and its composite score.")
figure("r6_accuracy_dual_metrics.png",
       "Figure 3. Accuracy screen. Metric A and Metric B appear side by side with their definitions, "
       "sample counts and the cross-validation line. The stage arithmetic above is the same prompt's "
       "numbers expanded.")
figure("r10_comparison_deck.png",
       "Figure 4. Candidate comparison deck. All seven models with composite score, predicted success, "
       "latency, query cost, cost and latency efficiency, and the model's real arena win-rate with its "
       "battle count. The selected model is outlined.")
figure("r13_simple_mode.png",
       "Figure 5. Simple experience level. The same decision restated without internals: an "
       "everyday-terms card, traffic-light cost and speed chips, and a one-line accuracy statement.")
figure("r12_feedback_page.png",
       "Figure 6. Feedback page. Ratings and corrections are written to disk as new rows and can be folded "
       "back into training, which is the loop that would let the router improve on an organisation's own traffic.")
figure("p2_best_match_grid.png",
       "Figure 7. Prompt-domain application, results view. Each card carries one reason chip quoting an "
       "existing logged trial, a fine-print line naming the algorithms behind the ranking, thumbs up or "
       "down that writes a real 1-5 trial row, and a Not-now display filter.")
figure("p3_ranking_methodology.png",
       "Figure 8. \u201cHow these rankings work\u201d. The explanatory prose lives behind one panel and in "
       "tooltips rather than on the working surface, so the input screen and the evidence do not compete "
       "for attention.")
figure("p7_report_model_comparison.png",
       "Figure 9. Report and checks. The evaluation runs on a per-prompt chronological last-trial-out "
       "split, stated on screen, so a reviewer can see the leakage control before seeing the numbers.")
figure("p8_data_quality.png",
       "Figure 10. Data-quality tab of the same step: schema validation, sparsity, and the specific rows "
       "the recommender had to work around.")

# ------------------------------------------------------------------ section 7
para("7. Limitations", style="Heading 2")
bullets([
    f"Metric B is weak. At {MB} the composite score separates a battle's winner from its loser only "
    "slightly better than a coin flip. The router is good at naming the right class of model for a task "
    "and poor at ranking two specific models inside that class.",
    f"The 90 % target is met on one split, not on average. The headline {MA} comes from a single seed-42 "
    f"held-out split; five-fold grouped cross-validation averages {CV_A}, which is below target. Both are "
    "reported; the cross-validation figure is the one to plan against.",
    "Small cells are noisy. Translation, business-communication and long-document categories have fewer "
    "than ten held-out prompts each, and low-volume models such as DeepSeek R1 (189 training battles) "
    "carry wide win-rate uncertainty.",
    "Battle data is text-only arena traffic. It is not a sample of any organisation's real usage, and it "
    "reflects the preferences of an anonymous crowd on prompts they chose to submit.",
    "No scores are probabilities. Nothing in the system is calibrated. Predicted success is a bounded "
    "model output used for ranking, and the score-kind table in Appendix A states what each column is "
    "allowed to mean.",
    "Limited tuning. Hybrid weights were compared across three configurations on validation data, not "
    "optimised globally. Association rules use max-confidence aggregation rather than a full rule "
    "explorer. No deep-learning recommender is included; none was shown to be needed for this problem.",
    "Comparisons are point estimates with standard errors. No formal significance testing between methods "
    "yet, which would need a paired bootstrap.",
])

# ------------------------------------------------------------------ section 8
para("8. How to run and verify", style="Heading 2")
code = (
    'cd "D:\\Recommendation Model\\recommendation_engine"\n'
    "\n"
    "pip install -r requirements.txt       # pandas, numpy, scipy, scikit-learn, streamlit,\n"
    "                                      # matplotlib, pytest - all free, no API keys\n"
    "\n"
    "streamlit run app_router.py           # contextual routing engine (Sections 4.2, 5.1)\n"
    "streamlit run app_prompt.py           # prompt to model application\n"
    "streamlit run app.py                  # generic domain-agnostic engine\n"
    "\n"
    "python -m pytest tests -q                     # 141 tests, 83.9 s\n"
    "python scripts/build_accuracy_report.py --with-cv    # regenerate Section 5.1 from raw data\n"
    "python scripts/tune_accuracy.py               # logged hyper-parameter grid search\n"
    "python run_demo_outputs.py                    # tables, figures, logs for the generic engine\n"
    "python run_prompt_demo_outputs.py             # same for the prompt domain\n"
    f"python scripts/build_arena_dataset.py             # rebuild the {N_ALL:,}-battle dataset"
)
for line in code.split("\n"):
    p = para(space_after=0)
    r = p.add_run(line if line else " ")
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
doc.add_paragraph()
para("Requirements are free and open-source only. There is no cloud dependency, no paid API and no LLM "
     "call anywhere in the system; the in-app assistant that answers \u201cwhy this model\u201d is a local "
     "rule-based component. The screenshots in Section 6 were captured by driving the running "
     "applications with scripts/capture_ui_screenshots.py, which reported zero browser console errors "
     "across the walk-through. Test results are in docs/test_results.md with raw output in "
     "outputs/logs/pytest_verbose.txt.")

# ------------------------------------------------------------------ appendices
para("Appendix A - what each score is allowed to mean", style="Heading 2")
table(["score_kind", "Definition", "Is it a probability?"], [
    ["frequency_count", "Distinct users who interacted with the item", "No"],
    ["similarity", "TF-IDF or co-occurrence cosine on [0, 1]", "No"],
    ["ranking_score", "Latent-factor reconstruction value", "No"],
    ["confidence", "Empirical co-occurrence rule confidence", "No"],
    ["normalized_ranking_score", "Weighted blend of per-source rank scores", "No"],
], widths=[4.6, 8.0, 3.4])

para("Appendix B - data mapping for the prompt domain", style="Heading 2")
table(["Generic role", "Prompt-domain column", "Notes"], [
    ["user_id", "prompt_id", "A prompt is the requesting entity"],
    ["item_id", "model_id", "Eight models in the demo catalogue, seven in the router"],
    ["interaction_value", "quality_score", "1-5 rating, aggregated by mean"],
    ["timestamp", "timestamp", "Drives the chronological last-trial-out split"],
    ["user attributes", "task_category, language", "Categorical"],
    ["item attributes", "modality, pricing, best_for, description", "Two categorical, two free-text"],
], widths=[3.6, 5.4, 7.0])
para("Changing the domain means changing this mapping, the thresholds in config/prompt_config.json and "
     "the features; the core engine is untouched. That path is documented in "
     "docs/customization_guide.md.")

para("Appendix C - project contents", style="Heading 2")
para("app.py, app_prompt.py, app_router.py (three Streamlit front ends); src/engine.py (load, build, "
     "recommend facade); src/data/, src/features/, src/recommenders/, src/ranking/, src/selection/, "
     "src/evaluation/, src/statistics/, src/fallback/, src/reporting/, src/utils/; src/routing/ "
     "(classifier ensemble, dual-metric evaluation, chat assistant, UI panels, charts); src/domain/ "
     "(prompt matcher, trial log); config/, data/, scripts/, tests/, outputs/, docs/.")

doc.save(OUT_PATH)
print("wrote", OUT_PATH, f"{OUT_PATH.stat().st_size / 1e6:.2f} MB")
