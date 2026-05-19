# Full 3Y Historical Rating Validation

```
Historical ratings validation summary
- sampled rows: 25
- checked no-snapshot strategies: final_quant_21d_no_snapshot, final_quant_21d_no_snapshot_sector_capped, final_quant_21d_no_snapshot_with_sma_filter, final_quant_5d_no_snapshot, final_quant_5d_no_snapshot_loose, final_quant_5d_no_snapshot_no_sma_filter, final_quant_63d_no_snapshot, final_quant_63d_no_snapshot_sector_capped, final_quant_63d_no_snapshot_with_sma200_filter, final_quant_model_1y_no_snapshot, final_quant_model_no_snapshot
- validated snapshot fields excluded from final_quant_model_1y_no_snapshot: all_time_avg_price_target, all_time_target_count, all_time_target_upside, analyst_count, consensus_target, consensus_upside, high_target, high_target_upside, last_month_avg_price_target, last_month_target_count, last_month_target_upside, last_quarter_avg_price_target, last_quarter_target_count, last_quarter_target_upside, last_year_avg_price_target, last_year_target_count, last_year_target_upside, low_target, low_target_upside, median_target, target_revision_30d, target_revision_7d, target_spread
- failures: 0
- status: PASS
- no sampled row showed evidence of using a grades-historical record after the feature date
- snapshot models are labeled snapshot_current
- historical rating-count models are labeled with historical analyst data modes
- missing historical rating-count data is filled safely
```
