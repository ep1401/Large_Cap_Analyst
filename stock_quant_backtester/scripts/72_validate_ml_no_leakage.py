from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.research_models import ML_ALLOWED_FEATURES, ML_TARGET_COLUMN, RESEARCH_STRATEGY_FEATURES, load_ml_artifact
from src.scoring import SNAPSHOT_FIELD_COLUMNS
from src.utils import load_dataframe


def main() -> None:
    runtime = Config.from_env()
    failures: list[str] = []

    artifact_path = runtime.project_root / "models" / "ml_ranker_no_snapshot.pkl"
    if not artifact_path.exists():
        failures.append(f"Missing ML artifact: {artifact_path}")
    else:
        artifact = load_ml_artifact(artifact_path)
        if artifact.get("train_end_date") != "2024-12-31":
            failures.append("ML artifact train_end_date must be 2024-12-31")
        if artifact.get("validation_start_date") != "2025-01-01":
            failures.append("ML artifact validation_start_date must be 2025-01-01")
        if artifact.get("validation_end_date") != "2025-12-31":
            failures.append("ML artifact validation_end_date must be 2025-12-31")
        if set(artifact.get("feature_names", [])) != set(ML_ALLOWED_FEATURES):
            failures.append("ML artifact feature_names do not match the allowed no-snapshot ML feature list")
        if ML_TARGET_COLUMN in set(artifact.get("feature_names", [])):
            failures.append("ML target column must not appear in the ML feature matrix")

    offending_ml_features = sorted(set(ML_ALLOWED_FEATURES) & SNAPSHOT_FIELD_COLUMNS)
    if offending_ml_features:
        failures.append(f"ML feature list contains snapshot columns: {', '.join(offending_ml_features)}")
    offending_future_features = sorted(feature for feature in ML_ALLOWED_FEATURES if feature.startswith("future_"))
    if offending_future_features:
        failures.append(f"ML feature list contains future-return columns: {', '.join(offending_future_features)}")

    for strategy_name, fields in RESEARCH_STRATEGY_FEATURES.items():
        offending = sorted(set(fields) & SNAPSHOT_FIELD_COLUMNS)
        if offending:
            failures.append(f"{strategy_name} uses snapshot fields: {', '.join(offending)}")

    features_path = runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv"
    if not features_path.exists():
        features_path = runtime.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    if not features.empty:
        train_rows = features.loc[(features["date"] >= "2023-01-01") & (features["date"] <= "2024-12-31")].copy()
        validation_rows = features.loc[(features["date"] >= "2025-01-01") & (features["date"] <= "2025-12-31")].copy()
        leaked_rows = features.loc[features["date"] >= "2026-01-01"].copy()
        if train_rows.empty:
            failures.append("No 2023-2024 rows available for ML training validation.")
        if validation_rows.empty:
            failures.append("No 2025 rows available for ML validation.")
        if not leaked_rows.empty and artifact_path.exists():
            artifact = load_ml_artifact(artifact_path)
            predictions = artifact.get("predictions_2025")
            if predictions is not None:
                prediction_dates = set(load_dataframe(runtime.tables_dir / "ml_ranker_no_snapshot_predictions_2025.csv", parse_dates=["date"])["date"].dt.year.unique())
                if any(year >= 2026 for year in prediction_dates):
                    failures.append("ML validation predictions include 2026 rows.")

    lines = [
        "# ML No Leakage Validation",
        "",
        "- This is a research candidate workflow.",
        "- 2026 forward data was not used for ML training or model selection.",
        "- Back-tested performance is hypothetical unless actually paper-tracked live.",
        "- Snapshot analyst target fields are excluded.",
        "- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "- News sentiment depends on Alpha Vantage coverage and classification.",
        "- ML models may overfit and require future forward validation.",
        "- This is research/paper trading only, not financial advice.",
        f"- Pass/fail: {'PASS' if not failures else 'FAIL'}",
        f"- Target column: `{ML_TARGET_COLUMN}`",
        f"- Feature count checked: {len(ML_ALLOWED_FEATURES)}",
    ]
    if failures:
        lines.append(f"- Findings: {' | '.join(failures)}")
    else:
        lines.append("- Findings: none")

    report_path = runtime.reports_dir / "ml_no_leakage_validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
