# Failures, fixes and fallback log

Honest record of what actually failed during the build, what was changed,
and what remains. (Runtime fallback events for each run go to
`outputs/logs/fallback_log.jsonl`.)

## Development failures (reproduced → root-caused → fixed)

| # | Symptom | Root cause | Fix | Verified by |
|---|---|---|---|---|
| 1 | `SyntaxError` importing popularity module | malformed f-string `#{rank ...}` | corrected expression | smoke run + suite |
| 2 | `ValueError: Input must be 1- or 2-d` in item-similarity suitability | `np.diag` called on a scipy sparse result | `.toarray()` before dense pair counting | test_recommenders |
| 3 | `TypeError: unexpected keyword 'random_seed'` | TruncatedSVD uses `random_state` | renamed kwarg | smoke + CF tests |
| 4 | Item-similarity "similarity" scores could exceed 1.0 | weighted sum unnormalised | divide by total seed weight | bounded-score test |
| 5 | CF returned empty for every warm demo user | demo users consume their whole taste group; full-rank reconstruction ≈ exact fit | components capped to 8 (configurable); documented as honest data-structure effect, not patched with fake signal | split tests show CF scores held-out cells |
| 6 | Hybrid scores ~0.006 for top items (order plausible, magnitudes meaningless) | min-max normalization included already-seen high scorers, compressing candidates | candidate-relative normalization (filter then normalize) | `test_hybrid_dedup_and_bounds` + demo outputs |
| 7 | App crash on "Config & fallback status" page (`StreamlitAPIException`) | `st.dataframe()` given a plain string | explicit info-branch | `test_app_sidebar_navigation` |
| 8 | App crash on recommendation click (`ImportError: tabulate`) | `df.to_markdown()` needs optional `tabulate` | replaced with `st.dataframe` (compatible alternative, no new dependency) | `test_core_recommendation_flow` |
| 9 | Leakage test failed on U007 despite correct code | test compared train max-time to *all rows sharing the held-out item id*, not to the user's held-out *event* | test rewritten to the real invariant (train ≤ earliest held-out event per user) | `test_no_leakage_train_precedes_heldout` |
| 10 | Association deemed eligible for single-item baskets | suitability lacked a co-occurrence-information check | added max-basket-size ≥ 2 requirement (eligibility defect found by the test) | `test_insufficient_volume_disables_cf_and_baskets` |
| 11 | Malformed-row test expected 2 dropped, validation dropped 3 | test's own fixture had 3 bad rows; expectation wrong, code right | corrected test expectation | suite |
| 12 | `AttributeError: 'str' object has no attribute 'name'` while listing enabled algorithms in the domain output script | assumed `engine.recommenders` was a list of objects; it is a `dict[name, object]` | read `engine.status.eligible` instead | `outputs/logs/prompt_demo_run_summary.json` regenerated |
| 13 | On the sample prompt data a chat assistant outranks image generators for an image prompt | not a code defect: every placeholder model mean is exactly 4.00, so the popularity source degenerates to trial counts | added an explicit hard model-type candidate filter in the domain UI + documented the artifact | `test_model_type_constraint_is_hard_filter`, `test_app_prompt_smoke` |
| 14 | Collaborative filtering missing from the domain evaluation table | train split leaves 21 interactions (< `min_total_interactions` 30) | left as-is: eligibility guard behaving correctly, documented in `docs/domain_prompt_recommendation.md` | `{'collaborative': 'only 21 interactions (< 30)'}` reported by the selection layer |
| 15 | Simplified UI showed "Built from: p, o, p, u, l, a, r, i, t, y, ;, …" | `Recommendation.as_dict()` already joins `sources` with `;`, the app then joined that string again, splitting it into characters | split on `;` in the row adapter before rendering | `test_query_renders_comparison_cards_with_real_reasons` asserts readable source lists |
| 16 | Redesigned step 1 crashed on load: `StreamlitAPIException: selection_mode must be one of ['single','multi']` | Streamlit 1.55 no longer accepts `selection_mode=None` for `st.pills` | use `selection_mode="single"` | `test_step1_opens_on_a_search_plus_browse_front_page` boots the page |
| 17 | Windows console raised `UnicodeEncodeError` while printing a success message containing `→` | Git Bash Python stdout used cp1252, not a defect in the app (the browser renders it) | verification script re-encoded output as ASCII; app left unchanged | manual headless run after the change |
| 18 | In-app trial logger first allocated ids from the prompt file only | deleting a prompt row could make a new prompt reuse an id that existing trials already referenced | `next_prompt_id` now takes the maximum across `my_prompts.csv` **and** `my_trials.csv` | `test_ids_never_reuse_after_deletions_of_higher_numbers` |
| 19 | Featured UI pass: `st.pills` chips looked right but could not be driven headlessly | Streamlit 1.55 `AppTest` does not expose pills (`at.get("pills")` is empty and the widget returns `None` outside a browser), so a filter built from pills cannot be asserted | pills stay decorative (example prompts); every functional filter is a `selectbox`/`checkbox` that AppTest can set | `test_model_type_filter_is_applied`, `test_quick_tap_writes_a_real_trial_row` |
| 20 | The carousel seed button appeared not to fill the prompt box | verification-script error, not an app defect: `AppTest` `.click()` only records the click, the script must be re-run with `at.run()` before reading elements | helpers always click then `at.run()`; seeding confirmed via a staging key plus `st.rerun()` so the value is set before the `text_area` is instantiated | `test_search_this_prompt_loads_the_logged_text_into_the_box`, `test_browse_button_returns_to_the_discovery_panel` |
| 21 | Hiding models could leave an empty suggestion screen | marking every candidate **Not now** (or ticking "hide models I already rated" over a full pool) yields zero candidates, and `candidate_ids=[]` correctly means "nothing may appear" | quiet recovery line naming the controls that fix it, plus session **Undo n**; reproduced by hiding every image model | `test_hiding_every_candidate_recovers_and_undo_restores` |
| 22 | App-style pass: a segmented "filter tab" strip could not be driven headlessly | Streamlit 1.55 `AppTest` exposes neither `st.pills` nor `st.segmented_control` (both read back empty / `None` outside a browser), so a control built from them is untestable here | the filter strip stays `selectbox`/`checkbox` widgets and is collapsed behind **Refine ▾** instead of restyled as tabs | `test_filters_are_collapsed_until_refine_is_tapped`, `test_model_type_filter_is_applied` |
| 23 | New `.streamlit/config.toml` theme appeared to have no effect | Streamlit reads config at server start, so a running server keeps serving the old theme | restarted the port-8502 server; verified the live process reads `theme.base/primaryColor/…` and `client.toolbarMode="minimal"` via `streamlit config show`, plus HTTP 200 and `_stcore/health` = `ok` | boot check, `docs/test_results.md` |

Fallbacks invoked in demo runs: **none** — on the synthetic demo all five
algorithms pass suitability and fit cleanly (see
`outputs/logs/demo_run_summary.json`: `"algorithms_excluded": {}`). The prompt-domain
runs likewise record no fallback events (`outputs/logs/fallback_log.jsonl` does not
exist; `outputs/logs/prompt_demo_run_summary.json`: `"algorithms_excluded": {}`), and
the only algorithm drop-out ever seen was the *correct* eligibility exclusion in row 14.
Fallback machinery itself is proven by `tests/test_fallback.py` (crash →
popularity substitute → logged; correction attempt path; disabled path;
failing source cannot break the hybrid).

## Environment limitations encountered

- Browser-based visual QA of the Streamlit UI was blocked by environment
  policy; UI correctness rests on AppTest flows (boot, front page with hero
  search + browse carousel and its arrows, carousel seed into the box, query →
  ranked comparison cards, reason chips, collapsed-then-opened filter strip,
  hard model-type filter, hide/undo including the all-hidden recovery, one-tap
  rating written to the log, step 2 tabs, step 3 evaluation) plus an HTTP 200
  boot and `_stcore/health` check. The look itself - `.streamlit/config.toml`
  colours, rounded card chrome, chips, carousel dots - is verified only
  structurally (the config values are read by the running server, and the
  markup/classes are asserted); no pixel-level confirmation exists.
- `plotly`, `mlxtend` not installed by design (avoided: matplotlib charts and
  a native association implementation).
- Harmless joblib/loky CPU-probing warning on this Windows machine
  (`LOKY_MAX_CPU_COUNT` can silence it).
- Streamlit 1.55 deprecated `use_container_width`; both apps now pass
  `width="stretch"` (no deprecated kwargs remain in `app.py` or
  `app_prompt.py`).

## Remaining known issues

- Weight tuning explores only 3 configurations (documented, not global).
- No rolling-origin evaluation or paired significance tests yet.
- Diversity uses the first categorical item field only.
- Large catalogs (>3000 items) make item-similarity/association ineligible
  rather than chunked (feasibility guard, honest exclusion).
