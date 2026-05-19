from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, summarize_backtest
from src.research_models import (
    ML_ALLOWED_FEATURES,
    ML_TARGET_COLUMN,
    RESEARCH_CAVEAT_LINES,
    build_ml_estimators,
    build_scored_frames_from_predictions,
    compute_rank_spread_for_scores,
    fit_and_score_ml_model,
    load_ml_artifact,
    precompute_research_panels,
    run_low_turnover_research_backtest,
    save_ml_artifact,
)
from src.utils import load_dataframe, save_dataframe


TRAIN_START = "2023-01-01"
TRAIN_END = "2024-12-31"
VALIDATION_START = "2025-01-01"
VALIDATION_END = "2025-12-31"


def _load_research_features(runtime: Config) -> pd.DataFrame:
    full_path = runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv"
    if full_path.exists():
        return load_dataframe(full_path, parse_dates=["date"])
    return load_dataframe(runtime.final_dir / "features_panel.csv", parse_dates=["date"])


def _prepare_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = features.copy()
    working["date"] = pd.to_datetime(working["date"])
    required = ML_ALLOWED_FEATURES + [ML_TARGET_COLUMN, "future_5d_return", "future_5d_spy_return"]
    for column in required:
        if column not in working.columns:
            raise ValueError(f"Missing required column for ML training: {column}")
    working = working.loc[working["ticker"] != "SPY"].copy()
    working = working.dropna(subset=[ML_TARGET_COLUMN]).copy()
    train_df = working.loc[(working["date"] >= pd.Timestamp(TRAIN_START)) & (working["date"] <= pd.Timestamp(TRAIN_END))].copy()
    validation_df = working.loc[(working["date"] >= pd.Timestamp(VALIDATION_START)) & (working["date"] <= pd.Timestamp(VALIDATION_END))].copy()
    return train_df, validation_df


def _evaluate_validation_predictions(runtime: Config, predictions: pd.DataFrame, model_id: str) -> dict[str, object]:
    scored_frames = build_scored_frames_from_predictions(
        features=load_dataframe(runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv", parse_dates=["date"])
        if (runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv").exists()
        else load_dataframe(runtime.final_dir / "features_panel.csv", parse_dates=["date"]),
        predictions=predictions,
    )
    rank_stats = compute_rank_spread_for_scores(scored_frames)
    features_full = (
        load_dataframe(runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv", parse_dates=["date"])
        if (runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv").exists()
        else load_dataframe(runtime.final_dir / "features_panel.csv", parse_dates=["date"])
    )
    validation_features = features_full.loc[
        (features_full["date"] >= pd.Timestamp(VALIDATION_START)) & (features_full["date"] <= pd.Timestamp(VALIDATION_END))
    ].copy()
    panels = precompute_research_panels(
        validation_features,
        runtime,
        scoring_mode="ml_prediction",
        prediction_df=predictions[["date", "ticker", "predicted_score"]],
        start_date=VALIDATION_START,
        end_date=VALIDATION_END,
        rebalance_frequency_days=15,
        holding_period_days=5,
    )
    weekly, _ = run_low_turnover_research_backtest(
        panels,
        top_n=10,
        cost_bps=20.0,
        enter_rank=10,
        hold_rank=20,
        max_holding_days=21,
        rebalance_frequency_days=15,
        strategy_name=model_id,
        exposure_mode="full",
    )
    summary = summarize_backtest(weekly, 5, model_id) if not weekly.empty else {}
    return {
        "model_name": model_id,
        "rank_correlation": rank_stats["rank_correlation"],
        "top_decile_avg_forward_excess": rank_stats["top_decile_avg_excess"],
        "bottom_decile_avg_forward_excess": rank_stats["bottom_decile_avg_excess"],
        "top_minus_bottom_spread": rank_stats["top_minus_bottom_spread"],
        "top_10_strategy_return": float(summary.get("full_period_total_return", float("nan"))),
        "spy_return": float(weekly["spy_value"].iloc[-1] / 10000.0 - 1.0) if not weekly.empty else float("nan"),
        "excess_vs_spy": float(summary.get("full_period_excess_return_vs_spy", float("nan"))),
        "sharpe": float(summary.get("sharpe_ratio", float("nan"))),
        "max_drawdown": float(summary.get("max_drawdown", float("nan"))),
        "average_turnover": float(summary.get("average_turnover", float("nan"))),
        "validation_rebalance_periods": int(summary.get("number_of_rebalance_periods", 0)),
    }


def main() -> None:
    runtime = Config.from_env()
    features = _load_research_features(runtime)
    train_df, validation_df = _prepare_split(features)

    if train_df.empty or validation_df.empty:
        raise SystemExit("ML training requires non-empty 2023-2024 training data and 2025 validation data.")

    estimators = build_ml_estimators()
    results: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, object] = {}

    for model_name, estimator in estimators.items():
        predictions, fitted = fit_and_score_ml_model(model_name, estimator, train_df, validation_df)
        prediction_frames.append(predictions.copy())
        fitted_models[model_name] = fitted
        results.append(_evaluate_validation_predictions(runtime, predictions, model_name))

    results_df = pd.DataFrame(results).sort_values(
        ["excess_vs_spy", "top_minus_bottom_spread", "rank_correlation"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model_name"])
    best_predictions = predictions_df.loc[predictions_df["model_name"] == best_model_name].copy()
    artifact_path = runtime.project_root / "models" / "ml_ranker_no_snapshot.pkl"
    save_ml_artifact(artifact_path, fitted_models[best_model_name], best_model_name, best_predictions)

    save_dataframe(runtime.tables_dir / "ml_ranker_no_snapshot_results.csv", results_df)
    save_dataframe(runtime.tables_dir / "ml_ranker_no_snapshot_predictions_2025.csv", predictions_df)

    report_lines = [
        "# ML Ranker No Snapshot Report",
        "",
        *[f"- {line}" for line in RESEARCH_CAVEAT_LINES],
        "",
        f"- Train window: {TRAIN_START} to {TRAIN_END}",
        f"- Validation window: {VALIDATION_START} to {VALIDATION_END}",
        f"- Target column: `{ML_TARGET_COLUMN}`",
        f"- Feature count: {len(ML_ALLOWED_FEATURES)}",
        f"- Best validation model: `{best_model_name}`",
        f"- Best validation excess vs SPY: {fmt_pct(float(best_row['excess_vs_spy']))}",
        "",
        "## Validation Results",
        "",
        dataframe_to_markdown(
            results_df.assign(
                rank_correlation=results_df["rank_correlation"].round(4),
                top_decile_avg_forward_excess=results_df["top_decile_avg_forward_excess"].map(fmt_pct),
                bottom_decile_avg_forward_excess=results_df["bottom_decile_avg_forward_excess"].map(fmt_pct),
                top_minus_bottom_spread=results_df["top_minus_bottom_spread"].map(fmt_pct),
                top_10_strategy_return=results_df["top_10_strategy_return"].map(fmt_pct),
                spy_return=results_df["spy_return"].map(fmt_pct),
                excess_vs_spy=results_df["excess_vs_spy"].map(fmt_pct),
                sharpe=results_df["sharpe"].round(4),
                max_drawdown=results_df["max_drawdown"].map(fmt_pct),
                average_turnover=results_df["average_turnover"].round(4),
            )
        ),
        "",
        "## Notes",
        "",
        "- The ML artifact stores only the best 2025 validation model for later research comparison.",
        "- 2026 rows are excluded from training, validation, and model selection.",
    ]
    (runtime.reports_dir / "ml_ranker_no_snapshot_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved {runtime.tables_dir / 'ml_ranker_no_snapshot_results.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_ranker_no_snapshot_report.md'}")
    print(f"Saved {artifact_path}")


if __name__ == "__main__":
    main()
