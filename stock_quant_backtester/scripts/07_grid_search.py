from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.utils import load_dataframe, save_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    sliced = df.copy()
    if start is not None:
        sliced = sliced.loc[sliced["date"] >= start]
    if end is not None:
        sliced = sliced.loc[sliced["date"] <= end]
    return sliced


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *rows])


def main() -> None:
    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    analyst_data_is_point_in_time = not (
        "analyst_data_mode" in features.columns
        and features["analyst_data_mode"].fillna("").eq("snapshot_current").any()
    )

    strategy_names = [
        "full_model",
        "strict_checklist_model",
        "technical_only",
        "technical_momentum_model",
        "analyst_only",
    ]
    top_ns = [10, 20, 30]
    holding_periods = [5, 21, 63]
    analyst_thresholds = [5, 10, 15, 20]
    regime_filters = [False, True]
    regime_exposures = [0.0, 0.5]

    results: list[dict] = []

    for strategy_name, top_n, holding_period_days, analyst_threshold, use_regime_filter, regime_exposure in product(
        strategy_names,
        top_ns,
        holding_periods,
        analyst_thresholds,
        regime_filters,
        regime_exposures,
    ):
        effective_threshold = analyst_threshold if strategy_name != "technical_only" else 0
        weekly, _, diagnostics = run_weekly_backtest(
            features=features,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            top_n=top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            use_regime_filter=use_regime_filter,
            regime_exposure=regime_exposure,
            use_analyst_filters=(strategy_name != "technical_only"),
            analyst_count_threshold=effective_threshold,
            min_avg_dollar_volume=20_000_000,
            strategy_name=strategy_name,
            require_positive_revision_7d=(strategy_name == "strict_checklist_model"),
            resistance_window=63 if strategy_name == "strict_checklist_model" else 30,
        )

        full_metrics = calculate_performance_metrics(weekly, holding_period_days=holding_period_days)
        dev_metrics = calculate_performance_metrics(_slice_period(weekly, end=DEV_END), holding_period_days=holding_period_days)
        test_metrics = calculate_performance_metrics(_slice_period(weekly, start=TEST_START), holding_period_days=holding_period_days)
        avg_final_pass_count = (
            float(diagnostics["final_pass_count"].mean()) if not diagnostics.empty and "final_pass_count" in diagnostics.columns else 0.0
        )
        under_diversified_share = (
            float((diagnostics["selected_count"] < top_n).mean()) if not diagnostics.empty and "selected_count" in diagnostics.columns else 0.0
        )

        row = {
            "strategy_name": strategy_name,
            "top_n": top_n,
            "holding_period_days": holding_period_days,
            "analyst_count_threshold": effective_threshold,
            "use_regime_filter": use_regime_filter,
            "regime_exposure": regime_exposure,
            "analyst_data_is_point_in_time": analyst_data_is_point_in_time,
            "average_final_pass_count": avg_final_pass_count,
            "pct_periods_fewer_than_top_n": under_diversified_share,
        }
        row.update({f"full_{key}": value for key, value in full_metrics.items()})
        row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        by=["test_sharpe_ratio", "test_excess_total_return", "test_max_drawdown"],
        ascending=[False, False, False],
    )
    save_dataframe(config.tables_dir / "grid_search_results.csv", results_df)

    top_sharpe = results_df.head(10).round(6)
    top_excess = results_df.sort_values("test_excess_total_return", ascending=False).head(10).round(6)
    best_by_strategy = (
        results_df.sort_values(
            by=["strategy_name", "test_sharpe_ratio", "test_excess_total_return"],
            ascending=[True, False, False],
        )
        .groupby("strategy_name", as_index=False)
        .head(1)
        .round(6)
    )

    lines = [
        "# Grid Search Summary",
        "",
        f"- Evaluated configurations: {len(results_df)}",
        f"- {IMPORTANT_CAVEAT}",
        "- Results should be judged primarily on test-period performance, not full-period leaderboard position.",
        "",
        "## Top 10 By Test-Period Sharpe",
        "",
        _dataframe_to_markdown(top_sharpe),
        "",
        "## Top 10 By Test-Period Excess Return Vs SPY",
        "",
        _dataframe_to_markdown(top_excess),
        "",
        "## Best Configuration By Strategy",
        "",
        _dataframe_to_markdown(best_by_strategy),
    ]
    report_path = config.reports_dir / "grid_search_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved grid search results to {config.tables_dir / 'grid_search_results.csv'}")
    print(f"Saved grid search summary to {report_path}")


if __name__ == "__main__":
    main()
