# Forward Performance Monitor

- This is a frozen forward/out-of-sample monitoring report.
- No 2026 data should be used to retune the model.
- Current forward sample is short.
- Back-tested performance is hypothetical unless trades were actually paper-tracked live.
- This is research/paper trading only, not financial advice.

- Strategy name: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`
- Base score model: `final_quant_5d_weight_tuned_no_snapshot`
- Forward start date: 2026-01-02
- Latest date: 2026-04-22
- Latest feature date: 2026-05-18
- Sample note: Forward sample is still short and should not be used for retuning.

## Current Metrics

- Model return: -0.26%
- SPY return: 4.39%
- Excess return: -4.65%
- Model max drawdown: -12.06%
- SPY max drawdown: -5.86%
- Number of rebalance periods: 6
- Average turnover: 0.8667
- Estimated trading costs: 0.0104
- Average holdings: 10.00

## Current Book

- Current holdings: AMZN, NVDA, INTU, KO, SPGI, TMO, UNH, AMD, BAC, GOOGL
- Latest buys: UNH, AMD, BAC, GOOGL
- Latest sells: BKNG, NFLX, NOW, ORCL
- Latest holds: AMZN, NVDA, INTU, KO, SPGI, TMO

## Checkpoints

| checkpoint | checkpoint_date     | reached | model_return | spy_return | excess_return | model_drawdown | periods_beating_spy |
| ---------- | ------------------- | ------- | ------------ | ---------- | ------------- | -------------- | ------------------- |
| 3_months   | 2026-04-02 00:00:00 | True    | -5.91%       | -4.55%     | -1.37%        | -12.06%        | 0                   |
| 6_months   | 2026-07-02 00:00:00 | False   | -0.26%       | 4.39%      | -4.65%        | -12.06%        | 1                   |
| 9_months   | 2026-10-02 00:00:00 | False   | -0.26%       | 4.39%      | -4.65%        | -12.06%        | 1                   |
| 12_months  | 2027-01-02 00:00:00 | False   | -0.26%       | 4.39%      | -4.65%        | -12.06%        | 1                   |

## Paper Trading Decision Rules

- Do not retune until at least 12 months of forward data exist.
- If the model is still behind SPY after 12 months and has worse drawdown, mark the strategy as a failed forward test.
- If the model beats SPY after 12 months with similar or better drawdown, continue paper trading.
- If the model beats SPY by more than 5 percentage points after 12 months with acceptable drawdown and realistic costs, consider deeper live-trading due diligence, not immediate real-money deployment.

## Underperformance Attribution

- Attribution read: Underperformance looked broad-based rather than coming from only a few names.
- Top 5 contributors:
| ticker | total_contribution | average_weight | number_of_periods_held | average_return_while_held | contribution_to_excess_return |
| ------ | ------------------ | -------------- | ---------------------- | ------------------------- | ----------------------------- |
| AMD    | 3.87%              | 10.00%         | 1                      | 38.73%                    | 3.49%                         |
| ORCL   | 2.64%              | 10.00%         | 2                      | 13.18%                    | 2.07%                         |
| AMZN   | 2.63%              | 10.00%         | 2                      | 13.17%                    | 1.31%                         |
| LIN    | 1.20%              | 10.00%         | 2                      | 6.01%                     | 1.20%                         |
| AMAT   | 0.88%              | 10.00%         | 2                      | 4.42%                     | 1.11%                         |

- Bottom 5 contributors:
| ticker | total_contribution | average_weight | number_of_periods_held | average_return_while_held | contribution_to_excess_return |
| ------ | ------------------ | -------------- | ---------------------- | ------------------------- | ----------------------------- |
| TMO    | -2.36%             | 10.00%         | 4                      | -5.91%                    | -3.68%                        |
| SPGI   | -2.04%             | 10.00%         | 4                      | -5.10%                    | -3.36%                        |
| INTU   | -0.65%             | 10.00%         | 2                      | -3.27%                    | -1.98%                        |
| NOW    | -1.18%             | 10.00%         | 2                      | -5.88%                    | -1.74%                        |
| BAC    | -1.16%             | 10.00%         | 2                      | -5.80%                    | -1.69%                        |

## Historical Expectation Comparison

- Historical average excess per rebalance period: 0.56%
- 2026 excess per rebalance period so far: -1.36%
- Historical drawdown range across walk-forward windows: -13.79% to -2.75%
- 2026 drawdown so far: -12.06%

## Paper Tracking State

- Paper-trading history file present: True
- Latest paper-tracking date: 2026-04-22
- Live model value: $10,000.00
- Live SPY value: $10,000.00
- Note: initialized_from_forward_test_no_prior_live_tracking

## Outputs

- Monitor table: `outputs/tables/forward_performance_monitor.csv`
- Attribution table: `outputs/tables/forward_underperformance_attribution.csv`
- Equity curve: `outputs/charts/forward_performance_equity_curve.png`
- Excess return chart: `outputs/charts/forward_performance_excess_return.png`
- Drawdown chart: `outputs/charts/forward_performance_drawdown.png`