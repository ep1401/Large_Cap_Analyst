# ML 2026 Forward Summary

- This is a research candidate workflow.
- 2026 forward data was not used for ML training or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.

- Forward start date: 2026-01-02
- Latest available date: 2026-04-22
- Frozen candidate strategy: `ml_ranker_5d_no_snapshot`
- Frozen candidate model type: `hist_gradient_boosting_regression`
- Model path loaded from disk: `models/ml_ranker_no_snapshot.pkl`
- Execution mode: `low_turnover_hold_band`
- Enter rank: 10
- Hold rank: 20
- Top N: 10
- Max holding days: 21
- Rebalance frequency: 15 trading days
- Cost assumption: 20 bps

## Metrics

- ML return: 18.52%
- Rule-based model return: -0.26%
- SPY return: 4.39%
- ML excess vs SPY: 14.14%
- Rule-based excess vs SPY: -4.65%
- ML excess vs rule-based: 18.78%
- ML max drawdown: -4.13%
- Rule-based max drawdown: -12.06%
- SPY max drawdown: -5.86%
- ML average turnover: 1.133333
- Rule-based average turnover: 0.866667
- ML estimated trading costs: 0.0136
- Rule-based estimated trading costs: 0.0104
- ML average holdings: 10.00
- Rule-based average holdings: 10.00
- Rebalance periods: 6

## Latest ML Actions

- Current holdings: ABT, AMAT, AXP, BKNG, CAT, COST, CSCO, GE, INTU, WMT
- Latest buys: ABT, AXP, BKNG, CSCO, GE, INTU, WMT
- Latest sells: AVGO, DIS, JNJ, META, MSFT, NVDA, QCOM
- Latest holds: AMAT, CAT, COST

## Attribution

- Top 5 ML contributors: CAT, CSCO, AVGO, AMAT, NFLX
- Bottom 5 ML detractors: NOW, BKNG, QCOM, IBM, ACN
- Attribution read: The contribution profile was broad-based rather than dominated by only a few names.

## Overlap Vs Rule-Based

- Overlap tickers: INTU
- ML-only tickers: ABT, AMAT, AXP, BKNG, CAT, COST, CSCO, GE, WMT
- Rule-only tickers: AMD, AMZN, BAC, GOOGL, KO, NVDA, SPGI, TMO, UNH
- Holdings overlap ratio: 5.26%
- ML selecting materially different names: true

## Research Candidate Forward Status

- If ML beats SPY and rule-based model over 6+ months with acceptable drawdown, continue monitoring.
- If ML underperforms SPY, keep as research only.
- If ML beats SPY for 12 months and has acceptable drawdown/cost robustness, consider promotion later.
- Do not promote based on fewer than 6 months.