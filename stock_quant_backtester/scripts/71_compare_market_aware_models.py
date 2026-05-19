from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, summarize_backtest
from src.research_models import (
    RESEARCH_CAVEAT_LINES,
    build_scored_frames_from_predictions,
    compute_rank_spread_for_scores,
    load_ml_artifact,
    precompute_research_panels,
    run_low_turnover_research_backtest,
)
from src.utils import load_dataframe, save_dataframe


VALIDATION_START = "2025-01-01"
VALIDATION_END = "2025-12-31"


def _load_features(runtime: Config) -> pd.DataFrame:
    full_path = runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv"
    if full_path.exists():
        return load_dataframe(full_path, parse_dates=["date"])
    return load_dataframe(runtime.final_dir / "features_panel.csv", parse_dates=["date"])


def _evaluate_candidate(
    runtime: Config,
    features_2025: pd.DataFrame,
    *,
    strategy_name: str,
    scoring_mode: str,
    exposure_mode: str,
    prediction_df: pd.DataFrame | None = None,
) -> tuple[dict[str, object], list[pd.DataFrame]]:
    panels = precompute_research_panels(
        features_2025,
        runtime,
        scoring_mode=scoring_mode,
        prediction_df=prediction_df,
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
        strategy_name=strategy_name,
        exposure_mode=exposure_mode,
    )
    summary = summarize_backtest(weekly, 5, strategy_name) if not weekly.empty else {}
    scored_frames = [panel for _, panel, _, _ in panels]
    rank_stats = compute_rank_spread_for_scores(scored_frames)
    row = {
        "strategy_name": strategy_name,
        "exposure_mode": exposure_mode,
        "validation_total_return": float(summary.get("full_period_total_return", float("nan"))),
        "validation_spy_return": float(weekly["spy_value"].iloc[-1] / 10000.0 - 1.0) if not weekly.empty else float("nan"),
        "validation_excess_vs_spy": float(summary.get("full_period_excess_return_vs_spy", float("nan"))),
        "validation_sharpe": float(summary.get("sharpe_ratio", float("nan"))),
        "validation_max_drawdown": float(summary.get("max_drawdown", float("nan"))),
        "validation_average_turnover": float(summary.get("average_turnover", float("nan"))),
        "validation_windows_beating_spy": int(summary.get("windows_beating_spy", 0)),
        "validation_rebalance_periods": int(summary.get("number_of_rebalance_periods", 0)),
        "top_decile_avg_forward_excess": rank_stats["top_decile_avg_excess"],
        "bottom_decile_avg_forward_excess": rank_stats["bottom_decile_avg_excess"],
        "top_minus_bottom_spread": rank_stats["top_minus_bottom_spread"],
        "rank_correlation": rank_stats["rank_correlation"],
    }
    return row, scored_frames


def main() -> None:
    runtime = Config.from_env()
    features = _load_features(runtime)
    features_2025 = features.loc[
        (features["date"] >= pd.Timestamp(VALIDATION_START)) & (features["date"] <= pd.Timestamp(VALIDATION_END))
    ].copy()
    if features_2025.empty:
        raise SystemExit("No 2025 rows available for research comparison.")

    ml_artifact = load_ml_artifact(runtime.project_root / "models" / "ml_ranker_no_snapshot.pkl")
    ml_predictions = pd.DataFrame(ml_artifact["predictions_2025"]).copy()
    ml_scored_frames = build_scored_frames_from_predictions(features_2025, ml_predictions)

    rows: list[dict[str, object]] = []
    scored_frame_map: dict[str, list[pd.DataFrame]] = {}

    candidate_specs = [
        ("final_quant_5d_weight_tuned_low_turnover_no_snapshot", "base_tuned", "full", None),
        ("final_quant_5d_weight_tuned_market_regime_no_snapshot", "base_tuned", "discrete", None),
        ("final_quant_5d_weight_tuned_market_regime_continuous_no_snapshot", "base_tuned", "continuous", None),
        ("final_quant_5d_market_aware_score_no_snapshot", "market_aware", "full", None),
        ("ml_ranker_5d_no_snapshot", "ml_prediction", "full", ml_predictions),
        ("ml_ranker_5d_market_exposure_no_snapshot", "ml_prediction", "discrete", ml_predictions),
        ("ml_ranker_5d_market_exposure_continuous_no_snapshot", "ml_prediction", "continuous", ml_predictions),
    ]

    for strategy_name, scoring_mode, exposure_mode, prediction_df in candidate_specs:
        row, scored_frames = _evaluate_candidate(
            runtime,
            features_2025,
            strategy_name=strategy_name,
            scoring_mode=scoring_mode,
            exposure_mode=exposure_mode,
            prediction_df=prediction_df,
        )
        rows.append(row)
        scored_frame_map[strategy_name] = scored_frames

    comparison_df = pd.DataFrame(rows).sort_values(
        ["validation_excess_vs_spy", "validation_max_drawdown", "validation_average_turnover"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    spy_row = {
        "strategy_name": "SPY",
        "exposure_mode": "buy_hold",
        "validation_total_return": float(comparison_df["validation_spy_return"].iloc[0]) if not comparison_df.empty else float("nan"),
        "validation_spy_return": float(comparison_df["validation_spy_return"].iloc[0]) if not comparison_df.empty else float("nan"),
        "validation_excess_vs_spy": 0.0,
        "validation_sharpe": float("nan"),
        "validation_max_drawdown": float("nan"),
        "validation_average_turnover": 0.0,
        "validation_windows_beating_spy": 0,
        "validation_rebalance_periods": int(comparison_df["validation_rebalance_periods"].iloc[0]) if not comparison_df.empty else 0,
        "top_decile_avg_forward_excess": float("nan"),
        "bottom_decile_avg_forward_excess": float("nan"),
        "top_minus_bottom_spread": float("nan"),
        "rank_correlation": float("nan"),
    }
    comparison_df = pd.concat([comparison_df, pd.DataFrame([spy_row])], ignore_index=True)
    save_dataframe(runtime.tables_dir / "market_aware_model_comparison.csv", comparison_df)

    production_name = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
    production_row = comparison_df.loc[comparison_df["strategy_name"] == production_name].iloc[0]
    better_than_production = comparison_df.loc[
        (comparison_df["strategy_name"] != "SPY")
        & (comparison_df["strategy_name"] != production_name)
        & (comparison_df["validation_excess_vs_spy"] > production_row["validation_excess_vs_spy"])
    ].copy()

    candidate_lines = [
        "# Research Candidate Summary",
        "",
        *[f"- {line}" for line in RESEARCH_CAVEAT_LINES],
        "",
        f"- Current frozen production/paper model: `{production_name}`",
        "- 2026 remains excluded from training, feature selection, and model selection.",
        f"- Best 2025 research candidate by excess vs SPY: `{comparison_df.iloc[0]['strategy_name']}`",
        f"- Candidate(s) beating the current promoted model on 2025: {', '.join(better_than_production['strategy_name'].tolist()) if not better_than_production.empty else 'none'}",
        "- Any candidate listed here is still research-only and would require future forward validation before promotion.",
    ]
    (runtime.reports_dir / "research_candidate_summary.md").write_text("\n".join(candidate_lines), encoding="utf-8")

    report_lines = [
        "# Market Aware Model Comparison",
        "",
        *[f"- {line}" for line in RESEARCH_CAVEAT_LINES],
        "",
        "- Development window: 2023-01-01 to 2024-12-31",
        "- Validation window: 2025-01-01 to 2025-12-31",
        f"- Best ML validation model loaded from artifact: `{ml_artifact['model_name']}`",
        "",
        "## Comparison Table",
        "",
        dataframe_to_markdown(
            comparison_df.assign(
                validation_total_return=comparison_df["validation_total_return"].map(fmt_pct),
                validation_spy_return=comparison_df["validation_spy_return"].map(fmt_pct),
                validation_excess_vs_spy=comparison_df["validation_excess_vs_spy"].map(fmt_pct),
                validation_max_drawdown=comparison_df["validation_max_drawdown"].map(fmt_pct),
                validation_average_turnover=comparison_df["validation_average_turnover"].round(4),
                top_decile_avg_forward_excess=comparison_df["top_decile_avg_forward_excess"].map(fmt_pct),
                bottom_decile_avg_forward_excess=comparison_df["bottom_decile_avg_forward_excess"].map(fmt_pct),
                top_minus_bottom_spread=comparison_df["top_minus_bottom_spread"].map(fmt_pct),
                rank_correlation=comparison_df["rank_correlation"].round(4),
            )
        ),
        "",
        "## Research Readout",
        "",
        f"- Which model beats the current promoted model on 2025? {', '.join(better_than_production['strategy_name'].tolist()) if not better_than_production.empty else 'none'}",
        f"- Best drawdown among research candidates: `{comparison_df.dropna(subset=['validation_max_drawdown']).sort_values('validation_max_drawdown', ascending=False).iloc[0]['strategy_name']}`",
        f"- Lowest turnover among research candidates: `{comparison_df.dropna(subset=['validation_average_turnover']).sort_values('validation_average_turnover').iloc[0]['strategy_name']}`",
        f"- Best top-minus-bottom rank spread: `{comparison_df.dropna(subset=['top_minus_bottom_spread']).sort_values('top_minus_bottom_spread', ascending=False).iloc[0]['strategy_name']}`",
        f"- Does market sentiment/regime improve performance? {'yes' if not comparison_df.loc[comparison_df['strategy_name'].str.contains('market', na=False), 'validation_excess_vs_spy'].dropna().empty and comparison_df.loc[comparison_df['strategy_name'].str.contains('market', na=False), 'validation_excess_vs_spy'].max() > production_row['validation_excess_vs_spy'] else 'not yet'}",
        f"- Does ML improve performance? {'yes' if not comparison_df.loc[comparison_df['strategy_name'].str.contains('ml_ranker', na=False), 'validation_excess_vs_spy'].dropna().empty and comparison_df.loc[comparison_df['strategy_name'].str.contains('ml_ranker', na=False), 'validation_excess_vs_spy'].max() > production_row['validation_excess_vs_spy'] else 'not yet'}",
    ]
    (runtime.reports_dir / "market_aware_model_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved {runtime.tables_dir / 'market_aware_model_comparison.csv'}")
    print(f"Saved {runtime.reports_dir / 'market_aware_model_comparison.md'}")
    print(f"Saved {runtime.reports_dir / 'research_candidate_summary.md'}")


if __name__ == "__main__":
    main()
