from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.research_models import (
    ML_ALLOWED_FEATURES,
    ML_TARGET_COLUMN,
    load_ml_artifact,
    load_ml_research_candidate_config,
)
from src.scoring import SNAPSHOT_FIELD_COLUMNS
from src.utils import load_dataframe


def main() -> None:
    runtime = Config.from_env()
    candidate = load_ml_research_candidate_config(runtime.project_root)
    artifact_path = runtime.project_root / candidate.model_path
    failures: list[str] = []

    if not artifact_path.exists():
        failures.append(f"Missing ML artifact: {artifact_path}")
        artifact: dict[str, object] = {}
    else:
        artifact = load_ml_artifact(artifact_path)
        if artifact.get("model_name") != candidate.model_type:
            failures.append(
                f"Artifact model_name {artifact.get('model_name')} does not match config model_type {candidate.model_type}"
            )
        if str(artifact.get("train_end_date")) > "2024-12-31":
            failures.append("Artifact training window extends beyond 2024-12-31")
        if str(artifact.get("validation_end_date")) > "2025-12-31":
            failures.append("Artifact validation/model-selection window extends beyond 2025-12-31")
        if artifact.get("target_column") != ML_TARGET_COLUMN:
            failures.append(f"Artifact target_column must be {ML_TARGET_COLUMN}")
        feature_names = list(artifact.get("feature_names", []))
        if set(feature_names) != set(ML_ALLOWED_FEATURES):
            failures.append("Artifact feature list does not match the frozen no-snapshot ML feature list")
        offending_snapshot = sorted(set(feature_names) & SNAPSHOT_FIELD_COLUMNS)
        if offending_snapshot:
            failures.append(f"Artifact feature list includes snapshot fields: {', '.join(offending_snapshot)}")
        offending_future = sorted(feature for feature in feature_names if feature.startswith("future_"))
        if offending_future:
            failures.append(f"Artifact feature list includes future-return columns: {', '.join(offending_future)}")

    if candidate.strategy_name != "ml_ranker_5d_no_snapshot":
        failures.append("Frozen ML candidate must remain ml_ranker_5d_no_snapshot")
    if candidate.long_short:
        failures.append("Frozen ML candidate must remain long-only")
    if candidate.use_regime_filter:
        failures.append("Frozen ML candidate must keep regime filter off")
    if candidate.snapshot_fields_allowed:
        failures.append("Frozen ML candidate must keep snapshot fields disabled")

    features_path = runtime.final_dir / "features_panel_2026_forward.csv"
    if not features_path.exists():
        failures.append(f"Missing 2026 forward feature panel: {features_path}")
    else:
        features = load_dataframe(features_path, parse_dates=["date"])
        if not features.empty and pd_year_max(features) < 2026:
            failures.append("Forward feature panel does not contain 2026 rows")
        if any(column in SNAPSHOT_FIELD_COLUMNS for column in features.columns if column in ML_ALLOWED_FEATURES):
            failures.append("Forward feature panel exposes snapshot fields inside the ML feature matrix")

    predictions_path = runtime.tables_dir / "ml_ranker_no_snapshot_predictions_2025.csv"
    if not predictions_path.exists():
        failures.append(f"Missing 2025 validation predictions file: {predictions_path}")
    else:
        predictions = load_dataframe(predictions_path, parse_dates=["date"])
        if predictions.empty:
            failures.append("2025 validation predictions file is empty")
        else:
            min_year = int(predictions["date"].dt.year.min())
            max_year = int(predictions["date"].dt.year.max())
            if min_year < 2025 or max_year > 2025:
                failures.append("Validation predictions include dates outside 2025")

    artifact_mtime = (
        datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat(timespec="seconds")
        if artifact_path.exists()
        else "missing"
    )
    lines = [
        "# ML Forward No Leakage Validation",
        "",
        "- This is a research candidate workflow.",
        "- 2026 forward data was not used for ML training or model selection.",
        "- Back-tested performance is hypothetical unless actually paper-tracked live.",
        "- Snapshot analyst target fields are excluded.",
        "- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "- News sentiment depends on Alpha Vantage coverage and classification.",
        "- ML models may overfit and require future forward validation.",
        "- This is research/paper trading only, not financial advice.",
        "",
        f"- Pass/fail: {'PASS' if not failures else 'FAIL'}",
        f"- Candidate config: `{candidate.strategy_name}` / `{candidate.model_type}`",
        f"- Artifact path: `{candidate.model_path}`",
        f"- Artifact modified timestamp: {artifact_mtime}",
        f"- Training window cap checked: <= 2024-12-31",
        f"- Validation/model-selection window cap checked: <= 2025-12-31",
        "- 2026 model run must load the artifact from disk without retraining.",
    ]
    if failures:
        lines.append(f"- Findings: {' | '.join(failures)}")
    else:
        lines.append("- Findings: none")

    report_path = runtime.reports_dir / "ml_forward_no_leakage_validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


def pd_year_max(df):
    return int(df["date"].dt.year.max()) if not df.empty else -1


if __name__ == "__main__":
    main()
