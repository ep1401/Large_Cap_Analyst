# ML No Leakage Validation

- This is a research candidate workflow.
- 2026 forward data was not used for ML training or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.
- Pass/fail: PASS
- Target column: `future_5d_excess_return`
- Feature count checked: 28
- Findings: none