# ML Execution Interval Audit

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Conservative lag tests are used to detect possible timing leakage.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Strategy: `ml_ranker_5d_no_snapshot`
- Rebalance frequency: 15 trading days
- Max holding days: 21

- PASS: realized P&L used adjusted_close at the current and next decision dates for every held position in the 2026 forward window.
- PASS: SPY benchmark period compounding matched direct adjusted-close buy-and-hold within 0.0000%.