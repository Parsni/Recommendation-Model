"""Build DA 1, DA 2, and DA 3 native .docx submission reports.
"""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "outputs" / "screenshots" / "live_2026"
FIGS = ROOT / "outputs" / "figures"
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
# BUILD DA 1
# ==============================================================================
def build_da1():
    doc = setup_doc()
    para(doc, "Business Analytics  |  Recommendation Model Project", size=9.5, italic=True, color=RGBColor(0x66, 0x66, 0x66), space_after=2)
    para(doc, "Digital Assignment 1: Problem and Idea of the Model", style="Heading 1", space_after=3)
    para(doc, "Contextual LLM Routing Engine", size=13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79), space_after=2)
    para(doc, "A recommendation model that picks the right AI model for a prompt", size=10.5, italic=True, space_after=6)
    para(doc, "Name: Anirudh Ajai Pisharady and Shrimay Dhankar", size=9.5, space_after=2)
    para(doc, "Register number: 24BCE1805 and 24BCE1365", size=9.5, space_after=2)
    para(doc, "Programme: B.Tech Computer Science and Engineering, VIT Chennai", size=9.5, space_after=2)
    para(doc, f"Code Repository: {REPO_URL}", size=9.5, space_after=2)
    para(doc, f"Live Deployed App: {APP_URL}", size=9.5, bold=True, color=RGBColor(0x02, 0x84, 0xC7), space_after=2)
    para(doc, "Date: 31 October 2026", size=9.5, space_after=14)

    para(doc, "1. Title of the project", style="Heading 2")
    para(doc, "Prompt to Model Recommendation: A Contextual LLM Routing Engine\n"
              "The project recommends which AI model should answer a given prompt. It looks at the task, the user's priority and real human votes on past answers.")

    para(doc, "2. Introduction and background", style="Heading 2")
    para(doc, "AI chatbots are now part of daily work. People use them to write emails, fix code, solve maths problems and translate text. Many different models are on offer. GPT-4o, Claude, Gemini, DeepSeek, Llama and Qwen are a few of them.")
    para(doc, "These models are not the same. Each one has its own price, its own speed and its own strengths. A model that writes good stories may be weak at code. A cheap model may do well on a short email but fail on a proof.")
    para(doc, "Most users pick a model by habit. They stay with one favourite for everything. For one person this costs little. For a team that sends thousands of queries a month, it adds up. Sending every request to the best model is slow and expensive. Sending every request to the cheapest gives poor answers that must be redone.")
    para(doc, "Companies that run AI at scale call this choice routing. A router reads each request and sends it to the model that suits it best. This project builds a small router using methods from the Business Analytics syllabus.")
    para(doc, "We treat it as a recommendation problem. The prompt plays the role of the user. The AI model plays the role of the item. A human vote for one model's answer over another is the rating.")
    para(doc, "The seven models we plan to compare differ a lot in price. The table shows the prices set in our project configuration. They change often, so we treat them as inputs and not as facts about the market.")

    table(doc, ["Model", "Provider", "Input price (USD per 1M tokens)", "Output price (USD per 1M tokens)"], [
        ["GPT-4o", "OpenAI", "2.50", "10.00"],
        ["Claude 3.5 Sonnet", "Anthropic", "3.00", "15.00"],
        ["Gemini 1.5 Pro", "Google", "1.25", "5.00"],
        ["DeepSeek R1", "DeepSeek", "0.55", "2.19"],
        ["Qwen3 235B A22B", "Alibaba", "0.35", "1.40"],
        ["Llama 3.1 70B", "Meta", "0.35", "0.40"],
        ["GPT-4o Mini", "OpenAI", "0.15", "0.60"],
    ], widths=[3.5, 3.0, 4.8, 4.8])

    para(doc, "Output prices range from 0.40 to 15.00 dollars per million tokens. That is a gap of about 37 times. A router that avoids the costly model when it is not needed can save real money.")

    para(doc, "3. Problem statement", style="Heading 2")
    para(doc, "Given a written prompt and the user's priority (quality, cost or speed), recommend which of seven AI models should answer it.")
    para(doc, "The recommendation must meet three conditions:")
    bullets(doc, [
        "It must rest on real human preference data and not on opinion.",
        "It must show how the decision was made, step by step.",
        "It must report honestly how often it is right, on prompts it has never seen."
    ])

    para(doc, "4. Objectives of the model", style="Heading 2")
    bullets(doc, [
        "Build a recommender that ranks seven AI models for any prompt.",
        "Learn the strength of each model from thousands of real head to head votes.",
        "Combine answer quality with cost and speed. The user sets the weights.",
        "Explain each recommendation in plain steps that a non technical user can follow.",
        "Measure accuracy on held out data. Report two separate metrics and do not round them up.",
        "Reach at least 90 percent agreement with the best model per task category (Metric A).",
        "Beat a coin flip (50 percent) on picking the human winner in a single battle (Metric B).",
        "Provide a simple web app with a feedback page, so the model can improve with use."
    ])

    para(doc, "5. Proposed dataset and its source", style="Heading 2")
    para(doc, "We will use public data from LMArena (also called Chatbot Arena). On this site a person types a prompt. Two anonymous models reply. The person votes for the better reply. The vote is the signal we need.")
    para(doc, "We will combine two public releases from Hugging Face:")
    bullets(doc, [
        "lmarena-ai/arena-human-preference-100k",
        "lmarena-ai/arena-human-preference-140k (licence: CC BY 4.0)"
    ])
    para(doc, "Each row has the prompt, the two model names and the winner. We will keep only the rows that fit our needs.")

    table(doc, ["Filter step", "Rule"], [
        ["1", "English prompts only"],
        ["2", "Winner must be model A or model B. Ties and both bad votes are dropped."],
        ["3", "Both models must be one of our seven candidates"],
        ["4", "Prompt must have at least 10 characters"],
        ["5", "Exact duplicate rows are removed"],
    ], widths=[3.0, 13.0])

    para(doc, "After filtering we are left with 3,945 battles (3,245 from the 100k release and 700 from the 140k release). All of them are real votes. No synthetic rows are included.")
    para(doc, "Model specification table. Price per token and base latency for each model. We keep it in a configuration file.")
    para(doc, "Anchor prompts. A short list of example prompts for each task type. We use it to match a new prompt to a task category.")
    para(doc, "The prompts come from people who chose to use an online arena. They are not a sample of any one company's traffic. Some models, such as DeepSeek R1, have far fewer battles than GPT-4o. We will show these gaps in the report and not hide them.")

    para(doc, "6. Proposed methodology and statistical techniques", style="Heading 2")
    if (FIGS / "da1_planned_flow.png").exists():
        figure(doc, FIGS / "da1_planned_flow.png", "Figure 1. Planned flow from prompt to ranked recommendation.", width_cm=15.5)

    bullets(doc, [
        "Clean the prompt. Remove empty or unsafe text and measure its length, code density and maths density.",
        "Find the task type. Group prompts with K means clustering. Also compare the prompt to the anchor prompts using cosine similarity.",
        "Estimate quality. Fit a Bradley Terry model on the battle votes. It gives each model a strength score and a win probability for any pair.",
        "Estimate cost and speed. Use ridge regression to predict output length, latency and cost for each model.",
        "Score and rank. Combine the four values in one weighted score. Scale each to a 0 to 1 range first.",
        "Test. Split the data by prompt. Train on 80 percent. Score the other 20 percent. Repeat with five fold cross validation."
    ])
    para(doc, "Composite score = w1 x P(success) + w2 x similarity + w3 x cost efficiency + w4 x speed efficiency\n"
              "The default weights are 0.70, 0.15, 0.08 and 0.07. A user can switch to a cheaper or faster preset.")

    table(doc, ["Technique", "Where we use it"], [
        ["Descriptive statistics", "Battle counts, prompt length, spread and skew, task mix"],
        ["Probability and logistic regression", "Bradley Terry win probability for each model pair"],
        ["Confidence intervals", "Win rates and accuracy figures, so readers see the uncertainty"],
        ["Hypothesis testing", "Binomial test for accuracy against 50 and 90 percent. Chi square test for task type against winning model"],
        ["Effect size", "Cramer's V next to the chi square test"],
        ["Regression", "Ridge regression for tokens, latency and cost"],
        ["Clustering", "K means to group prompts into intent segments"],
        ["Similarity measures", "TF IDF and cosine similarity against anchor prompts"],
        ["Sampling and validation", "Group split by prompt and five fold cross validation"],
        ["Multi criteria decision analysis", "Min max scaling and weighted composite score"],
    ], widths=[6.5, 9.5])

    para(doc, "7. Expected recommendations and outcomes", style="Heading 2")
    bullets(doc, [
        "A ranked list of all seven models with a composite score for each.",
        "One clear top pick, with the reason in plain words.",
        "Predicted cost and response time for each model.",
        "A decision trail that shows what each stage computed.",
        "A page that reports accuracy, with sample sizes and the cross validation result.",
        "A feedback form. Ratings are saved so they can feed later retraining."
    ])

    table(doc, ["Type of prompt", "What we expect"], [
        ["Maths and logic", "A strong reasoning model, at a low price if its win rate is close"],
        ["Code", "Two or three models close together. The cheaper one wins when the user picks the cost preset"],
        ["Short factual questions", "A cheap and fast model is enough"],
        ["Creative writing and translation", "The models with the best win rate in those categories"],
    ], widths=[6.0, 10.0])

    bullets(doc, [
        "Metric A of at least 90 percent on held out data.",
        "Metric B above 50 percent. We expect only a small gain, because two good models often give similar answers.",
        "A clear list of weak spots, such as task types with very few examples."
    ])
    para(doc, "The project is useful for students, analysts and small teams. They can choose a model with evidence in place of habit, and they can see what each choice is likely to cost.")

    out = ROOT / "outputs" / "DA1_Problem_and_Idea_of_the_Model.docx"
    doc.save(out)
    print("wrote DA1:", out)


# ==============================================================================
# BUILD DA 2
# ==============================================================================
def build_da2():
    doc = setup_doc()
    para(doc, "Business Analytics  |  Recommendation Model Project", size=9.5, italic=True, color=RGBColor(0x66, 0x66, 0x66), space_after=2)
    para(doc, "Digital Assignment 2: Software and Model Implementation", style="Heading 1", space_after=3)
    para(doc, "Contextual LLM Routing Engine", size=13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79), space_after=2)
    para(doc, "A recommendation model that picks the right AI model for a prompt", size=10.5, italic=True, space_after=6)
    para(doc, "Name: Anirudh Ajai Pisharady and Shrimay Dhankar", size=9.5, space_after=2)
    para(doc, "Register number: 24BCE1805 and 24BCE1365", size=9.5, space_after=2)
    para(doc, "Programme: B.Tech Computer Science and Engineering, VIT Chennai", size=9.5, space_after=2)
    para(doc, f"Code Repository: {REPO_URL}", size=9.5, space_after=2)
    para(doc, f"Live Deployed App: {APP_URL}", size=9.5, bold=True, color=RGBColor(0x02, 0x84, 0xC7), space_after=2)
    para(doc, "Date: 31 October 2026", size=9.5, space_after=14)

    para(doc, "1. Brief introduction and objectives", style="Heading 2")
    para(doc, "This project recommends which AI model should answer a prompt. In DA 1 we proposed the idea. In this report we describe the working model.")
    para(doc, "The model is written in Python. It runs as a Streamlit web app (app_router.py). The full code is in the GitHub repository and in the zip file submitted with this report.")
    para(doc, "Objectives of the implementation:")
    bullets(doc, [
        "Rank seven AI models for any prompt using real human votes.",
        "Blend answer quality with predicted cost and speed.",
        "Show every stage of the decision to the user.",
        "Measure accuracy on prompts that were not used for training.",
        "Collect user feedback for later retraining."
    ])
    para(doc, "The seven models are GPT-4o, GPT-4o Mini, Claude 3.5 Sonnet, Gemini 1.5 Pro, DeepSeek R1, Llama 3.1 70B and Qwen3 235B A22B.")

    para(doc, "2. Dataset description and source", style="Heading 2")
    para(doc, "The training data is a file of real head to head votes from LMArena (Chatbot Arena). A person sends one prompt to two anonymous models and votes for the better answer. We built the file with scripts/build_arena_dataset.py from two public Hugging Face releases.")

    table(doc, ["Source", "Licence", "Rows scanned", "Rows kept"], [
        ["arena-human-preference-100k", "see dataset card", "106,134", "3,255"],
        ["arena-human-preference-140k", "CC BY 4.0", "135,634", "700"],
    ], widths=[5.5, 3.5, 3.5, 3.5])

    para(doc, "Ten exact duplicates were removed, so the final file has 3,945 battles. Each row holds the prompt, the two model names, the winner, an anonymous user id, the source and a few tags from LMArena.")

    table(doc, ["Model", "Battle appearances"], [
        ["GPT-4o", "2,300"],
        ["Claude 3.5 Sonnet", "2,016"],
        ["Gemini 1.5 Pro", "1,709"],
        ["Llama 3.1 70B", "611"],
        ["GPT-4o Mini", "602"],
        ["Qwen3 235B A22B", "417"],
        ["DeepSeek R1", "235"],
    ], widths=[8.0, 8.0])

    para(doc, "The gap between GPT-4o and DeepSeek R1 is large. We keep it in mind when reading win rates later.")
    para(doc, "Two small supporting files are used. One lists prices and base latency for each model. The other holds anchor prompts, which are example prompts for each task type.")

    para(doc, "3. Data preprocessing", style="Heading 2")
    para(doc, "The filters were applied in this order:")
    bullets(doc, [
        "Keep English prompts only.",
        "Drop ties and both bad votes. Only rows with a clear winner stay.",
        "Map every raw model name to one of the seven names. For example gpt-4o-2024-05-13 and chatgpt-4o-latest both become gpt-4o.",
        "Drop prompts shorter than 10 characters.",
        "Remove exact duplicates."
    ])
    para(doc, "After this there are no missing values in any column. We checked each column and found zero nulls.")
    para(doc, "Merging model versions is a choice with a cost. We cannot tell a May version from an August version of GPT-4o. We note this as a limit.")
    para(doc, "Every prompt typed into the app goes through a sanitizer first. It rejects text below 3 characters or above 10 MB. It also strips markdown tricks that try to inject instructions.")
    para(doc, "Prompt length is very uneven. The median is 120 characters but the longest is over 60,000. The distribution is strongly skewed to the right.")

    if (FIGS / "da2_prompt_length.png").exists():
        figure(doc, FIGS / "da2_prompt_length.png", "Figure 1. Prompt length across the 3,945 battles. Most prompts are short.", width_cm=14.0)

    table(doc, ["Statistic", "Prompt length (characters)"], [
        ["Mean", "761.7"],
        ["Median", "120"],
        ["Standard deviation", "2,808.7"],
        ["First quartile", "55"],
        ["Third quartile", "334"],
        ["Minimum", "10"],
        ["Maximum", "60,661"],
        ["Skewness", "8.4"],
    ], widths=[8.0, 8.0])

    para(doc, "The mean is far above the median. A few very long prompts pull it up. For this reason we use the median, and we use a log scale when we plot length.")
    para(doc, "A keyword and pattern detector (domains.py) tags each prompt with one of eight task categories. We use this tag to measure accuracy per task.")

    table(doc, ["Category", "Battles", "Share"], [
        ["General or mixed", "2,171", "55.0%"],
        ["Question answering", "604", "15.3%"],
        ["Code", "547", "13.9%"],
        ["Creative writing", "287", "7.3%"],
        ["Mathematics", "174", "4.4%"],
        ["Long documents", "73", "1.9%"],
        ["Translation", "50", "1.3%"],
        ["Business communication", "39", "1.0%"],
    ], widths=[6.0, 5.0, 5.0])

    para(doc, "More than half the prompts are general. Translation and business communication have fewer than 50 battles each. Results for these will be noisy.")

    para(doc, "4. Model methodology and implementation", style="Heading 2")
    table(doc, ["Folder or file", "Purpose"], [
        ["app_router.py", "Streamlit app with guided steps, decision trail and feedback page"],
        ["src/routing/sanitizer.py", "Cleans the prompt"],
        ["src/routing/featurizer.py", "Length, tokens, code and maths density, language"],
        ["src/routing/clustering.py", "TF IDF and K means intent segments"],
        ["src/routing/anchor_similarity.py", "Cosine similarity to anchor prompts"],
        ["src/routing/inference/", "Bradley Terry ensemble and ridge cost and latency models"],
        ["src/routing/scoring.py", "Min max scaling and composite score"],
        ["src/routing/evaluation.py", "Group split, Metric A, Metric B, cross validation"],
        ["src/routing/telemetry/", "Feedback storage and retraining loop"],
        ["config/routing_config.json", "Model prices, weights, thresholds"],
        ["scripts/", "Dataset build, accuracy report, tuning grid"],
    ], widths=[5.5, 10.5])

    para(doc, "Each prompt passes through six stages. The app shows the value each stage computed:")
    bullets(doc, [
        "Structural features. Length, estimated tokens, code density, maths density, script and question flag.",
        "Clustering. K means with 4 clusters (random seed 42) on TF IDF text features. Each prompt lands in an intent segment such as coding, creative, maths or general.",
        "Anchor similarity. Cosine similarity between the prompt and the anchor prompts. The nearest anchor sets the task category.",
        "Win probability. A Bradley Terry model is fitted on the battles. It gives each model a latent strength. The deployed ensemble is 85 percent Bradley Terry and 15 percent pointwise logistic. Probability is kept between 0.05 and 0.98, so no model is ever shown as certain.",
        "Cost and latency. Ridge regression predicts output tokens, latency and query cost for each model.",
        "Composite score. A weighted blend of the four values, each scaled to 0 to 1."
    ])
    para(doc, "Score = 0.70 x P(success) + 0.15 x similarity + 0.08 x cost efficiency + 0.07 x speed efficiency\n"
              "These are the default weights. The user can pick a different preset in step 2 of the app. A what if panel lets an expert change the weights and see the ranking move.")
    para(doc, "If the best model has a success probability below 0.40 or a composite score below 0.35, the app falls back to GPT-4o. It also notes that it did so.")
    para(doc, "A sidebar switch gives Simple, Standard or Expert view. Simple shows plain language and traffic light chips for cost and speed. Standard adds model internals. Expert adds settings and feedback data.")

    para(doc, "Public Live Deployment & Verification:", style="Heading 3")
    para(doc, f"The application is fully hosted on Streamlit Cloud and accessible at:\n{APP_URL}")
    para(doc, "To run locally:")
    code_text = "pip install -r requirements.txt\nstreamlit run app_router.py\npython scripts/build_accuracy_report.py --with-cv\npython -m pytest tests -q"
    for l in code_text.split("\n"):
        p = para(doc, space_after=0)
        r = p.add_run(l)
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
    doc.add_paragraph()
    para(doc, "The test suite has 141 tests. All 141 passed on the last recorded run (Python 3.10.11, about 84 seconds).")

    para(doc, "5. Statistical techniques used", style="Heading 2")
    table(doc, ["Technique", "Use in the model"], [
        ["Descriptive statistics", "Mean, median, SD, quartiles and skew of prompt length. Frequency table of task categories"],
        ["Bradley Terry model", "Turns pairwise votes into a strength score for each model"],
        ["Logistic regression", "Pointwise success prior, blended with Bradley Terry"],
        ["Beta(1,1) shrinkage", "Win rate per task category. Thin cells are pulled toward the model's overall win rate"],
        ["Ridge regression", "Predicts tokens, latency and cost"],
        ["K means clustering", "Intent segments"],
        ["Cosine similarity", "Prompt to anchor matching"],
        ["Min max scaling", "Puts price, latency and probability on one scale"],
        ["Group train test split", "No prompt is in both train and test"],
        ["Cross validation", "Five folds, grouped by prompt"],
    ], widths=[5.5, 10.5])

    para(doc, "We report two metrics. They answer different questions.")
    bullets(doc, [
        "Metric A, task category agreement. For each held out prompt, does the router name the model with the highest win rate in that prompt's category? The win rate is measured on training data only.",
        "Metric B, head to head agreement. In each held out battle, does the router score the human's winner above the loser? Chance is 50 percent."
    ])
    para(doc, "The split is by prompt. We train on 3,151 battles and test on 794. The seed is 42.")

    para(doc, "6. Screenshots and sample outputs", style="Heading 2")
    para(doc, "The screenshots below come from the running app. The sample prompt is: \"Draft a diplomatic corporate email requesting an update on a delayed vendor contract without sounding aggressive.\"")

    if (SHOTS / "r1_step1_prompt_and_templates.png").exists():
        figure(doc, SHOTS / "r1_step1_prompt_and_templates.png", "Figure 2. Steps 1 and 2. A prompt box with ten task templates, and a choice of optimisation priority.")
    if (SHOTS / "r4_winner_reveal.png").exists():
        figure(doc, SHOTS / "r4_winner_reveal.png", "Figure 3. The six stage trail and the selected model. Qwen3 235B A22B scored 0.7543.")
    if (SHOTS / "r5_stage_mathematics.png").exists():
        figure(doc, SHOTS / "r5_stage_mathematics.png", "Figure 4. Full calculation for the same prompt. Each stage shows its real values.")
    if (SHOTS / "r10_comparison_deck.png").exists():
        figure(doc, SHOTS / "r10_comparison_deck.png", "Figure 5. All seven models compared by score, success probability, latency, cost and real arena win rate.")
    if (SHOTS / "r6_accuracy_dual_metrics.png").exists():
        figure(doc, SHOTS / "r6_accuracy_dual_metrics.png", "Figure 6. Accuracy screen with Metric A, Metric B and the size of the held out set.")
    if (SHOTS / "r12_feedback_page.png").exists():
        figure(doc, SHOTS / "r12_feedback_page.png", "Figure 7. Feedback page. Ratings and corrections are saved for retraining.")

    para(doc, "Ranking table for the sample prompt:")
    table(doc, ["Rank", "Model", "Composite", "P(success)", "Latency", "Query cost"], [
        ["1", "Qwen3 235B A22B", "0.7543", "0.815", "575 ms", "$0.00014"],
        ["2", "GPT-4o Mini", "0.6393", "0.596", "321 ms", "$0.00006"],
        ["3", "DeepSeek R1", "0.6342", "0.695", "810 ms", "$0.00022"],
        ["4", "Llama 3.1 70B", "0.5327", "0.436", "282 ms", "$0.00006"],
        ["5", "GPT-4o", "0.5227", "0.541", "507 ms", "$0.00103"],
        ["6", "Gemini 1.5 Pro", "0.5131", "0.505", "600 ms", "$0.00051"],
        ["7", "Claude 3.5 Sonnet", "0.1603", "0.094", "684 ms", "$0.00150"],
    ], widths=[1.5, 4.5, 2.5, 2.5, 2.5, 2.5])

    para(doc, "We checked the top score by hand. 0.70 x 0.815 + 0.15 x 0.516 + 0.08 x 0.939 + 0.07 x 0.446 = 0.754. It matches the app.")

    para(doc, "7. Initial results and recommendations", style="Heading 2")
    table(doc, ["Measure", "Result", "Reference"], [
        ["Metric A, main held out split", "92.82%", "Target 90%. Met"],
        ["Metric B, main held out split", "54.28%", "Chance is 50%"],
        ["Metric A, mean of 5 folds", "88.08% (SD 4.77)", "Below 90% target"],
        ["Metric B, mean of 5 folds", "55.26% (SD 0.92)", "Chance is 50%"],
    ], widths=[6.0, 5.0, 5.0])

    bullets(doc, [
        "Tuning tried 36 settings of the regularisation strength, the Bradley Terry blend and the domain affinity scale. Metric A ranged from 90.81 to 92.82 percent. Metric B stayed between 54.28 and 54.79 percent. So the result is stable to these settings.",
        "The router picks the right type of model for a task most of the time.",
        "It is much weaker at choosing between two specific models in one battle. Metric B is only a little above chance.",
        "The 90 percent target is met on one split. It is not met on the five fold average. We plan around the 88 percent figure.",
        "Code is the weak large category. GPT-4o and DeepSeek R1 have almost the same win rate there (56.9 and 56.5 percent), so the pick is close to arbitrary.",
        "Translation, business communication and long documents have fewer than 10 test battles each. We do not read anything from their percentages.",
        "For maths, general questions and short factual tasks, use the router's pick. These categories have enough data.",
        "For code, treat GPT-4o and DeepSeek R1 as equals. Choose the cheaper one when cost matters.",
        "Do not use the numbers for small categories until more data is collected.",
        "Keep collecting user feedback. It will help the small categories most."
    ])

    out = ROOT / "outputs" / "DA2_Software_and_Model_Implementation.docx"
    doc.save(out)
    print("wrote DA2:", out)


if __name__ == "__main__":
    build_da1()
    build_da2()
