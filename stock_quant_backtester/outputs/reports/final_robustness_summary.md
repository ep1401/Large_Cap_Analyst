# Final Robustness Summary

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- The current model is cost-sensitive and should remain paper-trading only unless it survives realistic cost assumptions.
- This is research/paper trading only, not financial advice.

- Is the tuned model materially better than the old baseline? True.
- Does it survive higher cost assumptions? False.
- Did cost fragility improve? True.
- Does any tested variant beat SPY at 20 bps? True.
- Does any tested variant beat SPY at 30 bps? True.
- Is top_n=10 stable? True.
- Is performance concentrated in one period? True.
- Is it fragile to small scoring noise? False.
- Which signal groups actually matter? ratings_removed, events_removed, technical_removed, sentiment_removed, risk_penalty_removed.
- Is it ready for paper trading? True.
- Recommended trading configuration if improved: {'label': 'hold_band_15_21_20', 'holding_period_days': 5, 'full_period_total_return': 1.16472826501131, 'full_period_excess_return_vs_spy': 0.987957860811968, 'annualized_return': 1.2499535755717286, 'annualized_volatility': 0.2844886510169788, 'sharpe_ratio': 3.013742991971941, 'max_drawdown': -0.1379298816001868, 'average_turnover': 0.7791666666666667, 'average_holdings': 10.0, 'percent_periods_invested': 1.0, 'number_of_rebalance_periods': 48, '2024_h1_excess_return_vs_spy': 0.1651137823216795, '2024_h1_sharpe_ratio': 4.870693990147454, '2024_h2_excess_return_vs_spy': -0.0532481477404851, '2024_h2_sharpe_ratio': 0.8308260674985597, '2025_excess_return_vs_spy': 0.3453311432189649, '2025_sharpe_ratio': 3.306525308264884, 'windows_beating_spy': 2, 'walk_forward_average_excess_vs_spy': 0.1523989259333864, 'beats_spy_at_cost': nan, 'variant_name': nan, 'enter_rank': 10.0, 'hold_rank': 20.0, 'max_holding_days': 21.0, 'total_cost_bps': 20, 'average_holding_days': nan, 'break_even_cost_bps': nan, 'beats_spy_at_20_bps': nan, 'beats_spy_at_30_bps': nan, 'source_table': 'rebalance_frequency_test', 'variant_type': 'low_turnover_hold_band', 'rebalance_frequency_days': 15.0, 'max_turnover_per_rebalance': nan}
- What would make it ready for real-money testing? More live paper-trading history, better capacity/slippage evidence, and stronger robustness across additional out-of-sample windows.