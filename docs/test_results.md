# Test results (genuine run)

- Executed: 2026-10-19, `python -m pytest tests -v --color=no` on Python 3.10.11 / pytest 8.3.4 / win32
- Raw output preserved at `outputs/logs/pytest_verbose.txt`
- Result: **141 passed, 1 warning in 83.92s (0:01:23)**
- Scope: data ingestion/validation, all five algorithms + hybrid, eligibility, fallback,
  metrics, splits/leakage, statistics utilities, exports, both apps, the prompt -> AI-model
  domain, and the multi-criteria LLM routing engine: shared domain detection,
  dual-metric accuracy evaluation on 3,945 real LMArena battles (Metric A = 92.82 %,
  Metric B = 54.28 %, 5-fold CV A = 88.08 % ± 4.77), the offline rule-based chat
  assistant, dual-audience UI panels (`ui_panels` pure helpers), and AppTest flows for
  `app_router.py` (persistent decision trail, honest dual-metric section, charts,
  dedicated feedback page, Simple/Standard/Expert experience-level gating, and
  deterministic what-if sandbox re-ranking).
- Test breakdown: 103 pre-existing + 8 `test_routing_accuracy` + 9 `test_chat_assistant`
  + 12 `test_ui_panels` + 9 `test_app_router_smoke` = 141.

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.4, pluggy-1.6.0 -- C:\Users\Shrimay Dhankar\AppData\Local\Programs\Python\Python310\python.exe
cachedir: .pytest_cache
rootdir: D:\Recommendation Model\recommendation_engine
plugins: anyio-4.12.1, langsmith-0.8.3
collecting ... collected 141 items

tests/test_app_prompt_smoke.py::test_step1_opens_on_a_search_plus_browse_front_page PASSED [  0%]
tests/test_app_prompt_smoke.py::test_step1_uses_no_callout_boxes PASSED  [  1%]
tests/test_app_prompt_smoke.py::test_carousel_arrow_changes_the_visible_model PASSED [  2%]
tests/test_app_prompt_smoke.py::test_search_this_prompt_loads_the_logged_text_into_the_box PASSED [  2%]
tests/test_app_prompt_smoke.py::test_browse_button_returns_to_the_discovery_panel PASSED [  3%]
tests/test_app_prompt_smoke.py::test_query_renders_comparison_cards_with_real_reasons PASSED [  4%]
tests/test_app_prompt_smoke.py::test_filters_are_collapsed_until_refine_is_tapped PASSED [  4%]
tests/test_app_prompt_smoke.py::test_model_type_filter_is_applied PASSED [  5%]
tests/test_app_prompt_smoke.py::test_unmatched_prompt_states_the_popularity_fallback PASSED [  6%]
tests/test_app_prompt_smoke.py::test_not_now_hides_a_model_and_undo_restores_it PASSED [  7%]
tests/test_app_prompt_smoke.py::test_quick_tap_writes_a_real_trial_row PASSED [  7%]
tests/test_app_prompt_smoke.py::test_hiding_every_candidate_states_recovery_and_undo_restores PASSED [  8%]
tests/test_app_prompt_smoke.py::test_step2_tabs_show_history_board_and_logger PASSED [  9%]
tests/test_app_prompt_smoke.py::test_step3_evaluation_runs_and_explains_scores PASSED [  9%]
tests/test_app_router_smoke.py::test_all_chart_functions_render PASSED   [ 10%]
tests/test_app_router_smoke.py::test_router_launches_without_exceptions PASSED [ 11%]
tests/test_app_router_smoke.py::test_prompt_entry_and_pipeline_run PASSED [ 12%]
tests/test_app_router_smoke.py::test_continue_to_feedback_navigates PASSED [ 12%]
tests/test_app_router_smoke.py::test_sidebar_experience_level_exists PASSED [ 13%]
tests/test_app_router_smoke.py::test_standard_mode_shows_under_hood_and_plain_card PASSED [ 14%]
tests/test_app_router_smoke.py::test_expert_mode_exposes_hyperparams_and_export PASSED [ 14%]
tests/test_app_router_smoke.py::test_sandbox_cost_only_weights_flip_leader PASSED [ 15%]
tests/test_app_router_smoke.py::test_simple_mode_shows_plain_card_hides_under_hood PASSED [ 16%]
tests/test_app_smoke.py::test_app_launches_without_exceptions PASSED     [ 17%]
tests/test_app_smoke.py::test_app_sidebar_navigation PASSED              [ 17%]
tests/test_app_smoke.py::test_core_recommendation_flow PASSED            [ 18%]
tests/test_app_smoke.py::test_evaluation_flow_runs PASSED                [ 19%]
tests/test_chat_assistant.py::test_parse_intent_help_and_greeting PASSED [ 19%]
tests/test_chat_assistant.py::test_parse_intent_why_metrics_feedback PASSED [ 20%]
tests/test_chat_assistant.py::test_parse_intent_compare_detects_models PASSED [ 21%]
tests/test_chat_assistant.py::test_parse_intent_switch_policy_synonyms PASSED [ 21%]
tests/test_chat_assistant.py::test_parse_intent_new_prompt_fallback PASSED [ 22%]
tests/test_chat_assistant.py::test_respond_why_with_decision PASSED      [ 23%]
tests/test_chat_assistant.py::test_respond_switch_policy_returns_action PASSED [ 24%]
tests/test_chat_assistant.py::test_respond_metrics_without_report_is_safe PASSED [ 24%]
tests/test_chat_assistant.py::test_respond_new_prompt_routes PASSED      [ 25%]
tests/test_data_validation.py::test_demo_validation_detects_seeded_defects PASSED [ 26%]
tests/test_data_validation.py::test_prepared_counts_match_known_facts PASSED [ 26%]
tests/test_data_validation.py::test_original_files_not_mutated PASSED    [ 27%]
tests/test_data_validation.py::test_missing_required_column_reported PASSED [ 28%]
tests/test_data_validation.py::test_rows_with_missing_ids_dropped_and_recorded PASSED [ 29%]
tests/test_data_validation.py::test_invalid_interaction_value_coerced_not_crash PASSED [ 29%]
tests/test_data_validation.py::test_empty_dataset_handled PASSED         [ 30%]
tests/test_data_validation.py::test_unsupported_format_rejected PASSED   [ 31%]
tests/test_data_validation.py::test_metadata_duplicate_ids_deduplicated PASSED [ 31%]
tests/test_data_validation.py::test_insufficient_data_still_produces_structures PASSED [ 32%]
tests/test_evaluation.py::test_precision_recall_hit_known_values PASSED  [ 33%]
tests/test_evaluation.py::test_ndcg_known_values_binary PASSED           [ 34%]
tests/test_evaluation.py::test_ndcg_graded_gains PASSED                  [ 34%]
tests/test_evaluation.py::test_coverage_and_entropy PASSED               [ 35%]
tests/test_evaluation.py::test_chronological_split_matches_known_last_items PASSED [ 36%]
tests/test_evaluation.py::test_no_leakage_train_precedes_heldout PASSED  [ 36%]
tests/test_evaluation.py::test_train_excludes_heldout_pairs PASSED       [ 37%]
tests/test_evaluation.py::test_comparison_runs_and_is_reproducible PASSED [ 38%]
tests/test_exports.py::test_recommendations_to_df_columns PASSED         [ 39%]
tests/test_exports.py::test_save_and_reload_table PASSED                 [ 39%]
tests/test_exports.py::test_save_bar_chart_creates_png PASSED            [ 40%]
tests/test_fallback.py::test_fit_failure_falls_back_to_popularity PASSED [ 41%]
tests/test_fallback.py::test_no_alternative_disables_algorithm PASSED    [ 41%]
tests/test_fallback.py::test_single_correction_then_success PASSED       [ 42%]
tests/test_fallback.py::test_recommend_failure_returns_empty_and_logs PASSED [ 43%]
tests/test_fallback.py::test_failing_source_cannot_break_hybrid PASSED   [ 43%]
tests/test_prompt_domain.py::test_config_maps_domain_columns PASSED      [ 44%]
tests/test_prompt_domain.py::test_sample_data_integrity PASSED           [ 45%]
tests/test_prompt_domain.py::test_eligibility_on_prompt_data PASSED      [ 46%]
tests/test_prompt_domain.py::test_matcher_finds_same_category_neighbours PASSED [ 46%]
tests/test_prompt_domain.py::test_matcher_image_prompt_suggests_image_models PASSED [ 47%]
tests/test_prompt_domain.py::test_matcher_unrelated_prompt_falls_back_with_note PASSED [ 48%]
tests/test_prompt_domain.py::test_low_similarity_warning PASSED          [ 48%]
tests/test_prompt_domain.py::test_scores_are_labelled_not_probabilities PASSED [ 49%]
tests/test_prompt_domain.py::test_trialalled_models_are_not_filtered_out PASSED [ 50%]
tests/test_prompt_domain.py::test_ranking_is_deterministic PASSED        [ 51%]
tests/test_prompt_domain.py::test_model_type_constraint_is_hard_filter PASSED [ 51%]
tests/test_recommenders.py::test_popularity_known_order PASSED           [ 52%]
tests/test_recommenders.py::test_popularity_demo_top5_matches_known_facts PASSED [ 53%]
tests/test_recommenders.py::test_popularity_empty_data_raises PASSED     [ 53%]
tests/test_recommenders.py::test_item_similarity_recommends_paired_item PASSED [ 54%]
tests/test_recommenders.py::test_item_similarity_cold_user_empty PASSED  [ 55%]
tests/test_recommenders.py::test_content_based_matches_shared_words PASSED [ 56%]
tests/test_recommenders.py::test_content_based_requires_metadata PASSED  [ 56%]
tests/test_recommenders.py::test_collaborative_finds_block_neighbour PASSED [ 57%]
tests/test_recommenders.py::test_collaborative_insufficient_data_excluded PASSED [ 58%]
tests/test_recommenders.py::test_association_confidence_known_value PASSED [ 58%]
tests/test_recommenders.py::test_association_demo_rule_truth PASSED      [ 59%]
tests/test_routing_accuracy.py::test_detect_domain_buckets PASSED        [ 60%]
tests/test_routing_accuracy.py::test_domain_vocabulary_stable PASSED     [ 60%]
tests/test_routing_accuracy.py::test_group_split_by_prompt_no_leakage PASSED [ 61%]
tests/test_routing_accuracy.py::test_win_rate_tables_and_empirical_best PASSED [ 62%]
tests/test_routing_accuracy.py::test_overall_battle_win_rates_counts PASSED [ 63%]
tests/test_routing_accuracy.py::test_real_arena_dataset_shape PASSED     [ 63%]
tests/test_routing_accuracy.py::test_run_accuracy_evaluation_smoke PASSED [ 64%]
tests/test_routing_accuracy.py::test_precomputed_report_consistency PASSED [ 65%]
tests/test_routing_engine.py::test_sanitizer_valid_payload PASSED        [ 65%]
tests/test_routing_engine.py::test_sanitizer_injection_neutralization PASSED [ 66%]
tests/test_routing_engine.py::test_sanitizer_bounds_rejection PASSED     [ 67%]
tests/test_routing_engine.py::test_in_flight_featurizer_indicators PASSED [ 68%]
tests/test_routing_engine.py::test_intent_clusterer_fit_and_predict PASSED [ 68%]
tests/test_routing_engine.py::test_anchor_similarity_matrix PASSED       [ 69%]
tests/test_routing_engine.py::test_context_enrichment_bridge PASSED      [ 70%]
tests/test_routing_engine.py::test_classifier_ensemble PASSED            [ 70%]
tests/test_routing_engine.py::test_resource_regressors PASSED            [ 71%]
tests/test_routing_engine.py::test_multi_criteria_optimizer PASSED       [ 72%]
tests/test_routing_engine.py::test_reconciliation_failsafe_gate PASSED   [ 73%]
tests/test_routing_engine.py::test_end_to_end_engine_and_feedback PASSED [ 73%]
tests/test_selection_hybrid_engine.py::test_minmax_normalize_cases PASSED [ 74%]
tests/test_selection_hybrid_engine.py::test_metadata_absent_disables_content_based PASSED [ 75%]
tests/test_selection_hybrid_engine.py::test_insufficient_volume_disables_cf_and_baskets PASSED [ 75%]
tests/test_selection_hybrid_engine.py::test_hybrid_dedup_and_bounds PASSED [ 76%]
tests/test_selection_hybrid_engine.py::test_engine_end_to_end_demo PASSED [ 77%]
tests/test_selection_hybrid_engine.py::test_engine_candidate_filter_and_exclude_seen PASSED [ 78%]
tests/test_selection_hybrid_engine.py::test_unseen_zero_popularity_items_excluded_from_popularity_source PASSED [ 78%]
tests/test_statistics.py::test_describe_numeric_hand_checked PASSED      [ 79%]
tests/test_statistics.py::test_mean_ci_contains_mean_and_known_case PASSED [ 80%]
tests/test_statistics.py::test_proportion_ci_wald PASSED                 [ 80%]
tests/test_statistics.py::test_ttest_matches_scipy PASSED                [ 81%]
tests/test_statistics.py::test_mannwhitney_matches_scipy PASSED          [ 82%]
tests/test_statistics.py::test_paired_and_gof PASSED                     [ 82%]
tests/test_statistics.py::test_correlation_pearson_vs_spearman PASSED    [ 83%]
tests/test_statistics.py::test_categorical_association_caveat PASSED     [ 84%]
tests/test_statistics.py::test_outlier_iqr PASSED                        [ 85%]
tests/test_statistics.py::test_linear_regression_slope PASSED            [ 85%]
tests/test_statistics.py::test_forecast_errors_or_returns PASSED         [ 86%]
tests/test_statistics.py::test_cluster_small_data_refuses PASSED         [ 87%]
tests/test_trial_log.py::test_first_trial_creates_both_files PASSED      [ 87%]
tests/test_trial_log.py::test_same_prompt_reuses_id_and_forms_one_basket PASSED [ 88%]
tests/test_trial_log.py::test_ids_never_reuse_after_deletions_of_higher_numbers PASSED [ 89%]
tests/test_trial_log.py::test_invalid_inputs_rejected PASSED             [ 90%]
tests/test_trial_log.py::test_logged_trials_view PASSED                  [ 90%]
tests/test_trial_log.py::test_corrupt_existing_file_reports_the_column PASSED [ 91%]
tests/test_ui_panels.py::TestClassifyTier::test_cheap_vs_median PASSED   [ 92%]
tests/test_ui_panels.py::TestClassifyTier::test_fair_band PASSED         [ 92%]
tests/test_ui_panels.py::TestClassifyTier::test_pricy PASSED             [ 93%]
tests/test_ui_panels.py::TestClassifyTier::test_fast_slow_labels PASSED  [ 94%]
tests/test_ui_panels.py::TestClassifyTier::test_zero_median_fallback PASSED [ 95%]
tests/test_ui_panels.py::TestCostContextLine::test_free PASSED           [ 95%]
tests/test_ui_panels.py::TestCostContextLine::test_small_cost_mentions_questions_per_dollar PASSED [ 96%]
tests/test_ui_panels.py::TestCostContextLine::test_larger_cost_six_decimals_when_tiny PASSED [ 97%]
tests/test_ui_panels.py::TestLatencyLine::test_sub_second PASSED         [ 97%]
tests/test_ui_panels.py::TestLatencyLine::test_multi_second PASSED       [ 98%]
tests/test_ui_panels.py::TestPresetCaption::test_all_preset_keys_have_captions PASSED [ 99%]
tests/test_ui_panels.py::TestPresetCaption::test_unknown_preset_returns_empty PASSED [100%]

============================== warnings summary ===============================
tests/test_app_router_smoke.py::test_router_launches_without_exceptions
  C:\Users\Shrimay Dhankar\AppData\Local\Programs\Python\Python310\lib\site-packages\joblib\externals\loky\backend\context.py:136: UserWarning: Could not find the number of physical cores for the following reason:
  [WinError 2] The system cannot find the file specified
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 141 passed, 1 warning in 83.92s (0:01:23) ==================
```
