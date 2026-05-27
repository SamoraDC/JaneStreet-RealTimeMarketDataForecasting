# Report Scoreboard

## Scope

This report aggregates existing `candidate_summary.csv` files. It is a diagnostic index, not a new validation run and not a deployable selection policy.

## Filters

- `min_rows`: `10000000`
- `categories`: `stage3`
- `include_reports`: `all`
- `exclude_reports`: `none`

## Top Candidates

| category | report | candidate | family | rows | global_r2 | min_fold_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0144249686044 | 0.00804614860344 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0143825574476 | 0.00804491699318 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0143821316274 | 0.00804861524389 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | strong_oof_ridge_stack | strong_stack | 11028424 | 0.0143759761799 | 0.00812494724718 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | strong_oof_ridge_stack | strong_stack | 11028424 | 0.0143277286149 | 0.00811985996102 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | strong_oof_ridge_stack | strong_stack | 11028424 | 0.0143260786045 | 0.00812453306264 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | conservative_rls_prediction | strong_base | 11028424 | 0.0142749221935 | 0.00764705611516 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | conservative_rls_prediction | strong_base | 11028424 | 0.0142511564797 | 0.00766144055573 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | conservative_rls_prediction | strong_base | 11028424 | 0.0142431159492 | 0.00766083713861 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | strong_base | 11028424 | 0.0142203339493 | 0.00762738908338 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | strong_base | 11028424 | 0.0141947867875 | 0.00764144792835 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | strong_base | 11028424 | 0.014189005948 | 0.00764150093777 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_1_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141823946858 | 0.00776346637458 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141823946858 | 0.00776346637458 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_nonlinear_stage3_narrow_control_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141823946858 | 0.00776346637458 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_3_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141822362163 | 0.00777159406372 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141818271797 | 0.00770319635526 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_1_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141811066222 | 0.00775603591738 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_0_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141804173343 | 0.00769309536303 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_3_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141795212622 | 0.00776529166938 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_tb100_min20000_p1000_contextual_blend | contextual_blend | 11028424 | 0.0141766245764 | 0.0077768571953 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_2_w0p75_fixed_blend | fixed_blend | 11028424 | 0.014176457538 | 0.00776428518738 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_abs_diff_tb100_min20000_p1000_contextual_blend | contextual_blend | 11028424 | 0.0141762792459 | 0.00777491386176 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_2_w0p75_fixed_blend | fixed_blend | 11028424 | 0.0141752428893 | 0.00775670343536 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_tb100_min20000_p10000_contextual_blend | contextual_blend | 11028424 | 0.0141703426491 | 0.00776042588118 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_abs_diff_tb100_min20000_p10000_contextual_blend | contextual_blend | 11028424 | 0.0141701977531 | 0.00775958793085 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | wf_blend_0_wf_blend | walk_forward_blend | 11028424 | 0.0141698177627 | 0.00756408024594 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | wf_blend_0_wf_blend | walk_forward_blend | 11028424 | 0.0141678881064 | 0.00758585521576 |
| stage3 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | wf_blend_1_wf_blend | walk_forward_blend | 11028424 | 0.0141674071283 | 0.00772129792575 |
| stage3 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_abs_diff_tb100_min20000_p1000_contextual_blend | contextual_blend | 11028424 | 0.0141644212875 | 0.00775074376136 |

## Top By Category

| category | category_rank | report | candidate | global_r2 | min_fold_r2 |
| --- | --- | --- | --- | --- | --- |
| stage3 | 1 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0144249686044 | 0.00804614860344 |
| stage3 | 2 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0143825574476 | 0.00804491699318 |
| stage3 | 3 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0143821316274 | 0.00804861524389 |
| stage3 | 4 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | strong_oof_ridge_stack | 0.0143759761799 | 0.00812494724718 |
| stage3 | 5 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | strong_oof_ridge_stack | 0.0143277286149 | 0.00811985996102 |
| stage3 | 6 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | strong_oof_ridge_stack | 0.0143260786045 | 0.00812453306264 |
| stage3 | 7 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | conservative_rls_prediction | 0.0142749221935 | 0.00764705611516 |
| stage3 | 8 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | conservative_rls_prediction | 0.0142511564797 | 0.00766144055573 |
| stage3 | 9 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | conservative_rls_prediction | 0.0142431159492 | 0.00766083713861 |
| stage3 | 10 | strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | 0.0142203339493 | 0.00762738908338 |
| stage3 | 11 | strong_oof_subset_s23aux8_s17_gateway_batch_demean_zscore_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | 0.0141947867875 | 0.00764144792835 |
| stage3 | 12 | strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1 | gateway_risk_conservative_rls_abs_pred_s25_prediction | 0.014189005948 | 0.00764150093777 |
| stage3 | 13 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_1_w0p75_fixed_blend | 0.0141823946858 | 0.00776346637458 |
| stage3 | 14 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0141823946858 | 0.00776346637458 |
| stage3 | 15 | strong_oof_subset_s23aux8_s17_gateway_nonlinear_stage3_narrow_control_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0141823946858 | 0.00776346637458 |
| stage3 | 16 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_3_w0p75_fixed_blend | 0.0141822362163 | 0.00777159406372 |
| stage3 | 17 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0141818271797 | 0.00770319635526 |
| stage3 | 18 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_1_w0p75_fixed_blend | 0.0141811066222 | 0.00775603591738 |
| stage3 | 19 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_0_w0p75_fixed_blend | 0.0141804173343 | 0.00769309536303 |
| stage3 | 20 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_3_w0p75_fixed_blend | 0.0141795212622 | 0.00776529166938 |
| stage3 | 21 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_tb100_min20000_p1000_contextual_blend | 0.0141766245764 | 0.0077768571953 |
| stage3 | 22 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | fixed_blend_2_w0p75_fixed_blend | 0.014176457538 | 0.00776428518738 |
| stage3 | 23 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_abs_diff_tb100_min20000_p1000_contextual_blend | 0.0141762792459 | 0.00777491386176 |
| stage3 | 24 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | fixed_blend_2_w0p75_fixed_blend | 0.0141752428893 | 0.00775670343536 |
| stage3 | 25 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_tb100_min20000_p10000_contextual_blend | 0.0141703426491 | 0.00776042588118 |
| stage3 | 26 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_weight_abs_diff_tb100_min20000_p10000_contextual_blend | 0.0141701977531 | 0.00775958793085 |
| stage3 | 27 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | wf_blend_0_wf_blend | 0.0141698177627 | 0.00756408024594 |
| stage3 | 28 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1 | wf_blend_0_wf_blend | 0.0141678881064 | 0.00758585521576 |
| stage3 | 29 | strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1 | wf_blend_1_wf_blend | 0.0141674071283 | 0.00772129792575 |
| stage3 | 30 | strong_oof_contextual_blend_stage3_stream_narrow_v1 | ctx_blend_0_abs_diff_tb100_min20000_p1000_contextual_blend | 0.0141644212875 | 0.00775074376136 |

## Audit

- Reports scanned: `86`
- Candidate rows: `4157`
- Filtered rows: `2494`
- Skipped reports: `0`
- Status: `diagnostic_only`
