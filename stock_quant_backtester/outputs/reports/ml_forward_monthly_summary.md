# ML Forward Monthly Summary

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Strict leakage timing audits passed, but ML can still overfit.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Strategy: `ml_ranker_5d_no_snapshot`
- Forward months covered: 2026-01, 2026-02, 2026-03, 2026-04, 2026-05
- Latest month: 2026-05

## Monthly Returns

| month   | ml_return | spy_return | rule_based_return | ml_excess_vs_spy | ml_excess_vs_rule_based | current_drawdown | worst_drawdown | turnover |
| ------- | --------- | ---------- | ----------------- | ---------------- | ----------------------- | ---------------- | -------------- | -------- |
| 2026-01 | 0.021884  | 0.013993   | 7e-05             | 0.00789          | 0.021813                | 0.0              | 0.0            | 1.0      |
| 2026-02 | -0.04167  | -0.014262  | -0.057997         | -0.027408        | 0.016326                | -0.04167         | -0.04167       | 1.0      |
| 2026-03 | 0.047358  | -0.045008  | -0.067633         | 0.092367         | 0.114991                | -0.020716        | -0.020716      | 1.0      |
| 2026-04 | 0.125634  | 0.093597   | 0.069793          | 0.032037         | 0.055841                | 0.0              | 0.0            | 1.0      |
| 2026-05 | 0.028139  | 0.038582   | 0.059683          | -0.010443        | -0.031544               | 0.0              | 0.0            | 1.0      |

## Readout

- Latest month ML excess vs SPY: -1.04%
- Latest month ML excess vs rule-based: -3.15%
- Latest month helped forward result: false
- Current drawdown: 0.00%
- Worst drawdown so far: -4.17%
- Monthly turnover latest: 1.0000
- Overall attribution read: Performance looked broad-based rather than concentrated.
- Latest-month attribution read: Performance looked concentrated in a few names.
- Top contributors overall: CAT, CSCO, AVGO, AMAT, NFLX
- Bottom detractors overall: NOW, BKNG, QCOM, IBM, ACN
- Top contributors latest month: CSCO, COST, CAT, GE, WMT
- Bottom detractors latest month: BKNG, AXP, ABT, INTU, AMAT