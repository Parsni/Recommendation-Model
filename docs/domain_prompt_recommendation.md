# Domain adaptation: prompt → which AI model to use

Selected DA topic: *given a prompt, suggest which AI model/tool to run it on to get
the best result.* This document records how the generic engine was tailored for it,
what was actually run, and what the numbers on the bundled sample do and do not show.

## 1. The mapping

The generic data contract was reused unchanged; only the column mapping differs
(`config/prompt_config.json`):

| Generic concept | Domain meaning | Where it lives |
| --- | --- | --- |
| `user_id` | `prompt_id` (P001 … P021) | `data/prompt_rec/prompts.csv` |
| `item_id` | `model_id` (M_gpt, M_claude, …) | `data/prompt_rec/models.csv` |
| `interaction_value` | `quality_score` 1–5 you gave that model's output | `data/prompt_rec/trials.csv` |
| `timestamp` | when you ran the trial | `data/prompt_rec/trials.csv` |
| item metadata | what the model is `best_for`, `modality`, `pricing` | `models.csv` |
| user metadata | `task_category`, `language`, the `prompt_text` itself | `prompts.csv` |

A "recommendation" therefore means: *this model has earned high trial scores on
prompts that look like yours.*

## 2. What was added (nothing in the core was rewritten)

| File | Purpose |
| --- | --- |
| `config/prompt_config.json` | column mapping, `exclude_seen: false`, top_k 4, domain thresholds |
| `data/prompt_rec/generate_sample_data.py` | deterministic **placeholder** trial generator |
| `data/prompt_rec/{prompts,models,trials}.csv` | 21 prompts / 8 models / 63 trials |
| `data/prompt_rec/README_DATA_COLLECTION.md` | protocol for logging your own real trials |
| `src/domain/prompt_matcher.py` | cold-start adapter: TF-IDF over prompt text + similarity-weighted blend |
| `src/domain/trial_log.py` | append-only personal trial logger (writes new files only, guards id reuse) |
| `app_prompt.py` | domain UI in plain task language |
| `run_prompt_demo_outputs.py` | regenerates the tables/figure/log quoted below |
| `tests/test_prompt_domain.py`, `tests/test_app_prompt_smoke.py`, `tests/test_trial_log.py` | 31 domain tests |

## 3. Cold start for a brand-new prompt

A new prompt has no trials, so the engine's user-history path cannot fire.
`PromptMatcher` instead:

1. TF-IDF-embeds every logged `prompt_text`;
2. ranks the nearest logged prompts to your pasted text (cosine similarity);
3. asks the hybrid engine for each neighbour's model ranking;
4. accumulates `similarity × score` per model, deduplicates models, sorts;
5. writes explanations that **name the neighbour prompts that produced the result**
   (`P001 (sim 0.52: "…")`) — no invented reasons;
6. if nothing overlaps textually, it serves the popularity baseline and says so;
7. if the best similarity is below 0.15, it adds an uncertainty warning.

`Model type constraint` in the UI is a hard candidate filter on item metadata
(e.g. `modality == image`): it restricts the pool, it does not re-weight scores.

## 4. Genuine run output (placeholder data)

From `python run_prompt_demo_outputs.py`
(log: `outputs/logs/prompt_demo_run_summary.json`):

- 21 prompts, 8 models, 63 trials, matrix density 0.375
- engine build **42.8 ms**, mean per-query recommendation latency **7.07 ms**
- eligible algorithms on this data: association, collaborative, content_based,
  item_similarity, popularity (none excluded)
- every displayed score carries `score_kind = normalized_ranking_score` and an
  explicit "not a probability" limitation line.

Chronological last-out evaluation (`outputs/tables/prompt_eval_comparison.csv`),
K=4, 18 evaluated prompts, model weights untouched:

| method | precision@4 | hit_rate@4 | hit_rate_se | ndcg@4 | coverage@4 |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.1667 | 0.6667 | 0.1143 | 0.6052 | 0.750 |
| hybrid | 0.1667 | 0.6667 | 0.1143 | 0.5833 | 0.875 |
| content_based | 0.1944 | 0.7778 | 0.1008 | 0.4467 | 0.875 |
| item_similarity | 0.0 | 0.0 | 0.0 | 0.0 | 0.125 |
| association | 0.0 | 0.0 | 0.0 | 0.0 | 0.125 |

Collaborative filtering is **absent from the evaluation table by design**: after the
train split removes each prompt's two most recent trials only 21 interactions remain,
below `min_total_interactions = 30`, so the eligibility layer excluded it and reported
`{'collaborative': 'only 21 interactions (< 30)'}`. With your full real log this
threshold will pass and CF will be evaluated too.

## 5. What these numbers do NOT show

The bundled `quality_score` column is a deterministic pattern
(`3 + ((prompt_index + model_index) % 3)`) produced by
`data/prompt_rec/generate_sample_data.py`. **It is not benchmark data and no
statement about any real AI model's quality can be drawn from it.** Two visible
artifacts of that:

- every model's mean trial score is exactly 4.00
  (`outputs/tables/prompt_model_trial_stats.csv`), so "best average rated" is a tie;
  the popularity source then degenerates to *how many trials exist per model*
  (M_gpt 21, image models 3), which is why a chat assistant can outrank image
  generators unless you use the model-type constraint;
- item_similarity and association score 0.0 because each placeholder prompt was
  trialled on only three models — real logs with more trials per prompt give those
  algorithms something to work with.

These tables validate that the pipeline computes, ranks, explains, evaluates and
exports correctly. Effectiveness claims require your own logged trials.

## 6. Converting this into real DA findings

1. Pick 20–30 of your own prompts across the task categories you care about.
2. Run each on 3–4 candidate tools, rate the outputs 1–5 against the rubric in
   `data/prompt_rec/README_DATA_COLLECTION.md`, and record the timestamp.
3. Replace `trials.csv`, extend `prompts.csv` and `models.csv` (same headers).
4. `streamlit run app_prompt.py` → *My trial log → Data & uploads* → upload the three
   CSVs → rerun recommendations and *Report & checks → Model comparison*. (Or skip the
   uploads entirely: tap 👍 / 👎 on the suggestion cards, then press
   **Rank from my logged trials**.)
5. Report the real NDCG@K / hit-rate table, the model-by-category means, and the
   confidence intervals from the statistics layer; carry the limitations box
   (single rater, no inter-rater reliability check, small catalog) into the report.
6. Re-validate before drawing conclusions: with ≥ 4–5 trials per prompt the
   eligibility thresholds, CF and association rules all become live, and the
   validation split can finally be used for weight tuning (`do_tuning`).

## 7. Operating the domain app

Launch: `streamlit run app_prompt.py` (port 8502 while the generic app holds 8501).
Colours, fonts and the reduced toolbar come from `.streamlit/config.toml`, which
Streamlit reads at server start - restart the server after editing it.

Three guided steps in the sidebar; every step uses tabs so the input screen, the
evidence and the report numbers never compete on one scrolling page. The sidebar is
deliberately quiet: product name and one-line tagline, the step radio, chips
(session hides when there are any, plus the demo-data caveat while sample data is
loaded) and one muted counts line. No metric tiles, no algorithm list, no
doc-path text. Step 1 renders no `st.warning` / `st.info` callout box at all -
caveats are chips and fine print, and explanation lives in tooltips plus one
collapsed panel.

| Step | Layout | What you do there |
| --- | --- | --- |
| **🔍 Find a model** | hero question + one bordered search card (**Find my model**, **Refine ▾**, example chips), then either the **Recommended for you** browse panel (no query yet) or the **Best match for your prompt** grid | browse ranked models in a **horizontal carousel** (◀ ▶ plus dot position and `n of N`, one card per screen, fed by the most recent trial in the dataset), or paste a prompt to compare models in a two-column **card grid**: rank badge, name, modality/pricing chips, a relative-score bar labelled `x.xx ranking score`, one blue **reason chip** quoting the trial that produced the card, a muted `signals · …` source line, and a *Why this model* expander. Beside the heading: the echoed prompt as a quiet line, a match-strength chip (`Strong / Fair / Weak match · sim 0.43`) and a `n held back` chip. Below: collapsed *Evidence* table, CSV download with a `because` column, and **Rank from my logged trials** |
| **📝 My trial log** | *Ranked prompts* · *Model performance* · *Log a trial* · *Data & uploads* | inspect any logged prompt (your ratings plus the engine's ranking); read the model leaderboard (trials, mean rating, modality, best_for) and the category chart; **record new trials inside the app**; upload your own three CSVs |
| **📊 Report & checks** | *Model comparison* · *Data quality* · *Score guide & config* | run the leakage-safe chronological comparison (metrics, NDCG/hit-rate table, saved CSV, split strategy and limitations), read validation reports and the quality report (the demo-data warning survives only here), and see the plain-language score table (every row states `probability? = no`), active algorithms, weights, thresholds and seed |

### Discovery, reason chips and one-tap feedback

The step-1 screen follows the standard recommender-UI pattern: browse first,
compare on request, explain every card, and make feedback one tap - presented like
a store front page rather than a report.

- **Carousel for discovery, grid for comparison.** Before a query the screen
  shows a horizontal browse panel (`render_carousel`) with ◀ ▶ and a dot plus
  `n of N` read-out, one card per screen. Once a prompt is submitted the same
  card renderer is laid out two-per-row so models can be compared side by side.
  Nothing is behind a modal, dialog or popup; detail lives in expanders.
- **Inline feed signal.** The panel opens with `Because your most recent trial
  ran P0xx · <prompt text>` - a named row the reader can go and find - and
  **Search this prompt** moves that text into the input box without running it.
- **Filters stay collapsed.** Model type (hard catalogue filter), result count,
  the hide-seen toggle, **Undo n** and **Browse** live behind **Refine ▾**, so the
  default screen is search plus cards. They remain ordinary `selectbox` /
  `checkbox` widgets because `st.pills` and `st.segmented_control` are invisible to
  `AppTest` and so untestable here. If every candidate ends up filtered out the
  screen says so in one quiet line and points at the control that restores results.
- **Reason chip per card.** Each card carries one chip in exactly one of three
  honest forms: the neighbour trial that scored this model (`You rated it 5/5 on
  "…" · wording match 0.43`), the model's own history (`In your log: 9 trials,
  mean 4.00/5`), or, when nothing exists, `No trial row for this model - shown by
  catalogue metadata only`. Signals are built from `engine.data.interactions` rows
  and the TF-IDF neighbours already used for the ranking; there is no template that
  could state a reason the data does not support. Prompt-derived text is
  HTML-escaped before rendering, since it is user input. On the bundled sample the
  chips end with `· demo rating`.
- **Dense, unambiguous feedback.** 👍 / 👎 on any card writes a real trial row
  through `src/domain/trial_log.py` at the rubric anchors 4 and 2, appends to
  `uploaded/my_trials.csv`, confirms with a toast plus a quiet on-screen line, and
  immediately feeds the log counter, the CSV download and the *hide models I
  already rated* filter. **Not now** is deliberately weaker: a session-only display
  filter that hides the model, writes nothing, and is reversible with **Undo n**.
  The score guide in step 3 states the mapping and labels the hide control as a
  filter, not a learned preference.
- **Where the prose went.** `how_it_works()` is one collapsed *How these rankings
  work* expander holding the mechanics, the filter semantics, the tap mapping and
  the demo caveat, plus the active-algorithm line. Tooltips on each control carry
  the rest. Step 1 therefore keeps its honesty statements while showing none of
  them as a coloured box.

### Logging your own data inside the app

`Log a trial` writes through `src/domain/trial_log.py`:

- appends to **new** files `data/prompt_rec/uploaded/my_trials.csv` and
  `my_prompts.csv`; the bundled source CSVs are never modified;
- identical prompt text logged twice reuses one `prompt_id`, so two models rated on
  the same prompt form a comparison instead of two unrelated observations;
- ids come from the highest id ever issued across both files, so deleting a prompt
  cannot make a later prompt reuse an id that trials already reference;
- **Switch the app onto my log** rebuilds every step on your rows through the same
  column mapping. Re-verified on a one-row log against the current app: the
  eligibility layer correctly dropped collaborative filtering, item similarity and
  association and kept `content_based + popularity`, and the answer for that exact
  prompt shows the match chip `Strong match · sim 1.00`.

Answers persist between interactions, so switching tabs or steps does not lose
them; asking again replaces the previous answer, and **Browse** returns to the
discovery panel. Once your own log has rows, the answer screen also offers
**Rank from my logged trials**, the same switch action as in step 2, so the
collect → retrain loop closes without leaving step 1.
