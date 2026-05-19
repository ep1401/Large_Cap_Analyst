# ML Preprocessing Leakage Audit

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Conservative lag tests are used to detect possible timing leakage.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Candidate model type: `hist_gradient_boosting_regression`
- Artifact path: `models/ml_ranker_no_snapshot.pkl`

- PASS: artifact training max date is <= 2024-12-31.
- PASS: artifact validation/model-selection window is confined to 2025.
- PASS: loaded estimator from disk is a `Pipeline` with fitted preprocessing inside the saved pipeline.
- PASS: training code slices train and validation features separately before fitting.
- PASS: scaler/imputer/model are fit on train data and applied to validation through the saved pipeline.
- PASS: no full-panel fit_transform step was found in the ML training function.
- PASS: no explicit feature-selection stage is present in the training function.