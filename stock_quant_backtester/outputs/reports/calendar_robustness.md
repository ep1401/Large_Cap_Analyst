# Calendar Robustness

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

- Edge concentrated in one half-year/year: True.

| window_label | model_return | spy_return | excess_return | max_drawdown | number_of_rebalance_periods |
| ------------ | ------------ | ---------- | ------------- | ------------ | --------------------------- |
| 2023 H1      | 0.129        | 0.1724     | -0.0434       | -0.0562      | 25                          |
| 2023 H2      | 0.0284       | 0.0747     | -0.0463       | -0.1567      | 25                          |
| 2024 H1      | 0.3169       | 0.1689     | 0.148         | -0.0548      | 25                          |
| 2024 H2      | 0.0182       | 0.0791     | -0.0609       | -0.0845      | 26                          |
| 2025 H1      | 0.119        | 0.0604     | 0.0586        | -0.176       | 24                          |
| 2025 H2      | 0.18         | 0.1136     | 0.0664        | -0.0391      | 25                          |
| Full 2023    | 0.161        | 0.26       | -0.099        | -0.1567      | 50                          |
| Full 2024    | 0.3409       | 0.2614     | 0.0796        | -0.0845      | 51                          |
| Full 2025    | 0.3204       | 0.1809     | 0.1395        | -0.176       | 49                          |