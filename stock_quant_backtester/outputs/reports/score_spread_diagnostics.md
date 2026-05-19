# Score Spread Diagnostics

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

## Findings
- Higher score predicts higher next-5D return: True.
- Threshold is selecting genuinely better stocks: True.
- Average selected minus SPY spread: 0.11%.
- Average selected minus non-selected spread: 0.17%.
- Average top-decile minus bottom-decile spread: 0.06%.

## Best Months

| month   | selected_minus_spy | selected_minus_non_selected | top_decile_minus_bottom_decile | periods |
| ------- | ------------------ | --------------------------- | ------------------------------ | ------- |
| 2024-01 | 0.017882           | 0.019726                    | 0.029346                       | 5       |
| 2023-05 | 0.016526           | 0.016763                    | 0.035594                       | 4       |
| 2025-07 | 0.014087           | 0.023975                    | 0.029721                       | 5       |
| 2024-02 | 0.01354            | 0.014321                    | 0.009959                       | 4       |
| 2024-12 | 0.008121           | 0.011676                    | 0.015793                       | 4       |

## Weakest Months

| month   | selected_minus_spy | selected_minus_non_selected | top_decile_minus_bottom_decile | periods |
| ------- | ------------------ | --------------------------- | ------------------------------ | ------- |
| 2023-01 | -0.019173          | 0.0                         | 0.0                            | 4       |
| 2025-09 | -0.012777          | -0.014489                   | -0.025485                      | 4       |
| 2023-09 | -0.012565          | -0.013407                   | -0.019772                      | 4       |
| 2024-11 | -0.012499          | -0.012927                   | -0.009157                      | 4       |
| 2024-05 | -0.009963          | -0.00769                    | -0.00357                       | 4       |

## Monthly Summary

| month   | selected_minus_spy | selected_minus_non_selected | top_decile_minus_bottom_decile | periods |
| ------- | ------------------ | --------------------------- | ------------------------------ | ------- |
| 2023-01 | -0.019173          | 0.0                         | 0.0                            | 4       |
| 2023-02 | -0.004805          | -0.006399                   | -0.011252                      | 4       |
| 2023-03 | 0.005193           | 0.004747                    | 0.008677                       | 5       |
| 2023-04 | -0.001229          | 0.000999                    | -0.00167                       | 4       |
| 2023-05 | 0.016526           | 0.016763                    | 0.035594                       | 4       |
| 2023-06 | 0.002869           | 0.004758                    | 0.004673                       | 4       |
| 2023-07 | 0.004342           | 0.001824                    | 0.00845                        | 4       |
| 2023-08 | 0.005281           | 0.003181                    | -0.008852                      | 5       |
| 2023-09 | -0.012565          | -0.013407                   | -0.019772                      | 4       |
| 2023-10 | 0.0032             | 0.002918                    | 0.008371                       | 4       |
| 2023-11 | 0.003502           | 0.001277                    | 0.005651                       | 4       |
| 2023-12 | -0.005712          | -0.010033                   | 0.000229                       | 4       |
| 2024-01 | 0.017882           | 0.019726                    | 0.029346                       | 5       |
| 2024-02 | 0.01354            | 0.014321                    | 0.009959                       | 4       |
| 2024-03 | 0.003393           | 0.008167                    | 0.000322                       | 4       |
| 2024-04 | 0.002639           | 0.003718                    | 0.000927                       | 4       |
| 2024-05 | -0.009963          | -0.00769                    | -0.00357                       | 4       |
| 2024-06 | 0.007406           | 0.011368                    | -0.003627                      | 4       |
| 2024-07 | -0.001139          | -0.003996                   | -0.004688                      | 5       |
| 2024-08 | -0.005297          | -0.007448                   | -0.015627                      | 4       |
| 2024-09 | -0.003613          | -0.004445                   | -0.010731                      | 4       |
| 2024-10 | 0.001508           | 0.003738                    | 0.00796                        | 5       |
| 2024-11 | -0.012499          | -0.012927                   | -0.009157                      | 4       |
| 2024-12 | 0.008121           | 0.011676                    | 0.015793                       | 4       |
| 2025-01 | -0.004551          | -0.01053                    | 3.7e-05                        | 4       |
| 2025-02 | -0.002027          | -0.007613                   | -0.031605                      | 4       |
| 2025-03 | 0.002309           | 0.003147                    | -0.001278                      | 4       |
| 2025-04 | -0.005261          | -0.004594                   | 0.002527                       | 4       |
| 2025-05 | 0.00668            | 0.009894                    | 0.021328                       | 4       |
| 2025-06 | 0.004535           | 0.00292                     | -0.008716                      | 4       |
| 2025-07 | 0.014087           | 0.023975                    | 0.029721                       | 5       |
| 2025-08 | 0.00205            | -0.000591                   | -0.018804                      | 4       |
| 2025-09 | -0.012777          | -0.014489                   | -0.025485                      | 4       |
| 2025-10 | 0.002416           | 0.003684                    | 0.005483                       | 5       |
| 2025-11 | 0.001549           | 0.001427                    | -0.01107                       | 3       |
| 2025-12 | 0.00331            | 0.000974                    | -0.007277                      | 4       |