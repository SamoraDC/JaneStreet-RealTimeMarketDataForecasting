# Report Scoreboard

## Scope

This report aggregates existing `candidate_summary.csv` files. It is a diagnostic index, not a new validation run and not a deployable selection policy.

## Filters

- `min_rows`: `10000000`
- `categories`: `historical`
- `include_reports`: `all`
- `exclude_reports`: `none`

## Top Candidates

| category | report | candidate | family | rows | global_r2 | min_fold_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| historical | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | residual_tail | 11151360 | 0.0156301712023 | 0.0129099696947 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | residual_tail | 11151360 | 0.0156301712023 | 0.0129099696947 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | residual_tail | 11151360 | 0.0156301712023 | 0.0129099696947 |
| historical | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | wf_blend_5_wf_blend | walk_forward_blend | 11151360 | 0.0156230902638 | 0.0129110954337 |
| historical | strong_oof_hist_max1398_gateway_signed_square_cube_pair_v1 | fixed_blend_5_w0p5_fixed_blend | fixed_blend | 11151360 | 0.0156189638499 | 0.0128315223186 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_3_w0p25_fixed_blend | fixed_blend | 11151360 | 0.0156133134011 | 0.0129160573201 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p9_residual_tail | residual_tail | 11151360 | 0.0156087096025 | 0.0128935110938 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p95_residual_tail | residual_tail | 11151360 | 0.0156078765414 | 0.0128943545599 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p95_residual_tail | residual_tail | 11151360 | 0.0156078765414 | 0.0128943545599 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_5_w0p75_fixed_blend | fixed_blend | 11151360 | 0.0155977285694 | 0.0129122530666 |
| historical | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | wf_blend_3_wf_blend | walk_forward_blend | 11151360 | 0.0155964497691 | 0.0129209808832 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_1_w0p25_fixed_blend | fixed_blend | 11151360 | 0.0155940800581 | 0.0129137240074 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_3_w0p5_fixed_blend | fixed_blend | 11151360 | 0.0155918841313 | 0.0129209808832 |
| historical | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | residual_tail | 11151360 | 0.0155909312023 | 0.0129123318989 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | residual_tail | 11151360 | 0.0155909312023 | 0.0129123318989 |
| historical | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | residual_tail | 11151360 | 0.0155909312023 | 0.0129123318989 |
| historical | strong_oof_hist_max1398_gateway_signed_square_v1 | fixed_blend_5_w0p5_fixed_blend | fixed_blend | 11151360 | 0.0155882151572 | 0.0128781343911 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p99_residual_tail | residual_tail | 11151360 | 0.015587739864 | 0.0128858804995 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p99_residual_tail | residual_tail | 11151360 | 0.015587739864 | 0.0128858804995 |
| historical | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p9_residual_tail | residual_tail | 11151360 | 0.015586569867 | 0.0128861336054 |

## Top By Category

| category | category_rank | report | candidate | global_r2 | min_fold_r2 |
| --- | --- | --- | --- | --- | --- |
| historical | 1 | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | 0.0156301712023 | 0.0129099696947 |
| historical | 2 | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | 0.0156301712023 | 0.0129099696947 |
| historical | 3 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail | 0.0156301712023 | 0.0129099696947 |
| historical | 4 | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | wf_blend_5_wf_blend | 0.0156230902638 | 0.0129110954337 |
| historical | 5 | strong_oof_hist_max1398_gateway_signed_square_cube_pair_v1 | fixed_blend_5_w0p5_fixed_blend | 0.0156189638499 | 0.0128315223186 |
| historical | 6 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_3_w0p25_fixed_blend | 0.0156133134011 | 0.0129160573201 |
| historical | 7 | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p9_residual_tail | 0.0156087096025 | 0.0128935110938 |
| historical | 8 | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p95_residual_tail | 0.0156078765414 | 0.0128943545599 |
| historical | 9 | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p95_residual_tail | 0.0156078765414 | 0.0128943545599 |
| historical | 10 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_5_w0p75_fixed_blend | 0.0155977285694 | 0.0129122530666 |
| historical | 11 | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | wf_blend_3_wf_blend | 0.0155964497691 | 0.0129209808832 |
| historical | 12 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_1_w0p25_fixed_blend | 0.0155940800581 | 0.0129137240074 |
| historical | 13 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | fixed_blend_3_w0p5_fixed_blend | 0.0155918841313 | 0.0129209808832 |
| historical | 14 | strong_oof_hist_max1398_gateway_tail_fixed_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | 0.0155909312023 | 0.0129123318989 |
| historical | 15 | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | 0.0155909312023 | 0.0129123318989 |
| historical | 16 | strong_oof_hist_max1398_gateway_tail_wf_blend_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail | 0.0155909312023 | 0.0129123318989 |
| historical | 17 | strong_oof_hist_max1398_gateway_signed_square_v1 | fixed_blend_5_w0p5_fixed_blend | 0.0155882151572 | 0.0128781343911 |
| historical | 18 | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p99_residual_tail | 0.015587739864 | 0.0128858804995 |
| historical | 19 | strong_oof_hist_max1398_gateway_residual_tail_modes_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p99_residual_tail | 0.015587739864 | 0.0128858804995 |
| historical | 20 | strong_oof_hist_max1398_gateway_residual_tail_v1 | gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p9_residual_tail | 0.015586569867 | 0.0128861336054 |

## Audit

- Reports scanned: `68`
- Candidate rows: `3540`
- Filtered rows: `332`
- Skipped reports: `0`
- Status: `diagnostic_only`
