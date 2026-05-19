# ML Forward No Leakage Validation

- This is a research candidate workflow.
- 2026 forward data was not used for ML training or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.

- Pass/fail: PASS
- Candidate config: `ml_ranker_5d_no_snapshot` / `hist_gradient_boosting_regression`
- Artifact path: `models/ml_ranker_no_snapshot.pkl`
- Artifact modified timestamp: 2026-05-19T12:24:55
- Training window cap checked: <= 2024-12-31
- Validation/model-selection window cap checked: <= 2025-12-31
- 2026 model run must load the artifact from disk without retraining.
- Findings: none