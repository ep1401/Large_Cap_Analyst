# ML vs Rule-Based Forward Holdings

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
- Average overlap ratio: 6.24%

## Overlap By Rebalance Date

| date                | period_end_date     | ml_count | rule_count | overlap_count | overlap_ratio | shared_tickers | ml_only_tickers                                   | rule_only_tickers                                |
| ------------------- | ------------------- | -------- | ---------- | ------------- | ------------- | -------------- | ------------------------------------------------- | ------------------------------------------------ |
| 2026-01-02 00:00:00 | 2026-01-26 00:00:00 | 10       | 10         | 1             | 0.052632      | LIN            | ACN, AMAT, CAT, CSCO, GOOGL, MRK, NFLX, NOW, QCOM | ABBV, BA, BAC, MA, MSFT, NVDA, SPGI, TMO, V      |
| 2026-01-26 00:00:00 | 2026-02-17 00:00:00 | 10       | 10         | 1             | 0.052632      | GOOGL          | ACN, AVGO, CAT, CSCO, IBM, JNJ, MCO, MRK, NOW     | AMAT, BA, LIN, MA, MSFT, NVDA, SPGI, TMO, V      |
| 2026-02-17 00:00:00 | 2026-03-10 00:00:00 | 10       | 10         | 1             | 0.052632      | AVGO           | CRM, IBM, INTU, MCO, NFLX, NOW, ORCL, SPGI, TMO   | ABT, AMAT, GOOGL, KO, LLY, MA, META, V, WMT      |
| 2026-03-10 00:00:00 | 2026-03-31 00:00:00 | 10       | 10         | 2             | 0.111111      | AVGO, NOW      | AMAT, AMD, CAT, CVX, INTU, LIN, QCOM, SPGI        | BKNG, KO, LLY, MA, META, NFLX, ORCL, V           |
| 2026-03-31 00:00:00 | 2026-04-22 00:00:00 | 10       | 10         | 1             | 0.052632      | NVDA           | AMAT, AVGO, CAT, COST, DIS, JNJ, META, MSFT, QCOM | AMZN, BKNG, INTU, KO, NFLX, NOW, ORCL, SPGI, TMO |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | 10       | 10         | 1             | 0.052632      | INTU           | ABT, AMAT, AXP, BKNG, CAT, COST, CSCO, GE, WMT    | AMD, AMZN, BAC, GOOGL, KO, NVDA, SPGI, TMO, UNH  |

## ML-Only vs Rule-Only Performance

| bucket    | average_ticker_count | average_return | total_contribution | contribution_to_excess_return |
| --------- | -------------------- | -------------- | ------------------ | ----------------------------- |
| ml_only   | 8.833333             | 0.037089       | 0.200143           | 0.118507                      |
| shared    | 1.166667             | 0.007574       | -0.005359          | -0.010317                     |
| rule_only | 8.833333             | 0.003007       | 0.020629           | -0.061007                     |

## Readout

- Shared tickers ever held: AVGO, GOOGL, INTU, LIN, NOW, NVDA
- ML-only contribution to excess return: 11.85%
- Rule-only contribution to excess return: -6.10%
- ML is genuinely selecting better names: true