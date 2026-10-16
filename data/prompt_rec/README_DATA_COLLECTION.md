# Data collection protocol — prompt → AI-model recommender

The sample CSVs in this folder are **placeholder demonstrations**. For your
DA report you need a real dataset, and only you can produce it: your own
trial ratings of AI tools on your own prompts.

## Target schema (3 CSVs)

**trials.csv** — one row per trial (one prompt run on one model, one result judged):

| column | meaning |
|---|---|
| prompt_id | stable id, e.g. P001 |
| model_id | stable id, e.g. M_gpt |
| quality_score | your rating 1-5 (see rubric below) |
| timestamp | date of the trial, e.g. 2025-06-03 14:00:00 |

**prompts.csv** — the task catalogue ("users" in engine terms):

| column | meaning |
|---|---|
| prompt_id | matches trials |
| task_category | coding / creative_writing / image_generation / reasoning_math / translation / data_analysis / research_summary / ... |
| language | en, hi, ... |
| prompt_text | the exact prompt you pasted |

**models.csv** — the tool catalogue ("items"):

| column | meaning |
|---|---|
| model_id | matches trials |
| provider_label | friendly name |
| best_for | text: what it claims to specialise in |
| modality | text / image / code / multimodal |
| pricing | free / freemium / subscription / open source |

## Suggested collection plan (achievable in 1-2 weeks)

1. Write 20-30 prompts across 5-7 task categories you actually care about.
2. Pick 5-8 AI tools covering those modalities.
3. For each prompt, run it on 3-4 candidate tools (same prompt text, fresh
   chats, default settings).
4. Score 1-5 with a fixed rubric **before** seeing other tools' answers where
   possible:
   - 1 wrong/unusable · 2 flawed, needed rework · 3 acceptable with gaps
   - 4 good, minor edits · 5 excellent, used as-is
5. Log each trial as one row in trials.csv immediately (timestamped).

Aiming for ~60-120 trial rows gives enough volume for all algorithms plus
the evaluation split. Record which model version you used (models change -
state that in the report as a limitation).

## One-tap shortcut inside the app

Every suggestion card carries a **👍** and a **👎** tap (the tooltips name them
"Good fit" and "Poor fit"). A tap appends
a real trial row to `uploaded/my_trials.csv` using the rubric anchors above:
👍 = **4** (good, minor edits), 👎 = **2** (flawed, needed rework).

- the tap is your rating, recorded on the 1-5 scale - nothing is inferred for you;
- the prompt text logged is the one on screen; `task_category` is copied from a
  logged prompt only on a strong wording match (similarity >= 0.45), otherwise it
  is stored as `unspecified` rather than guessed;
- repeated taps on the same card append repeated rows (a re-rating is a new
  observation, and the chronological split reads them in order);
- **Not now** writes nothing: it only hides that model from the results on this
  screen for the current session. **Undo n** (in the **Refine ▾** strip) brings it
  back.

Use the taps to capture volume quickly, then confirm the important rows in
**My trial log → Log a trial** where you set the category and the exact 1-5 score.

## Loading your real data

Either overwrite these three CSVs (keeping the same column names) or upload
your own from **My trial log → Data & uploads**, or switch onto the in-app log
with one button (**My trial log → Log a trial → Switch the app onto my log**).
No config edit is needed if headers match; otherwise adjust
`config/prompt_config.json → column_mapping`.

Then re-run, per the revalidation protocol in `docs/customization_guide.md`:
validation counts → eligibility → chronological evaluation → weight tuning →
fallback drill, and cite the fresh artifacts in `outputs/`.
