from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import (
    run_condition_based_backtest,
    run_weekly_backtest,
    save_backtest_outputs,
    save_backtest_validation,
    save_filter_diagnostics,
)
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import strategy_analyst_data_mode
from src.plots import create_plots
from src.utils import LOGGER, load_dataframe, save_dataframe, str_to_bool


DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    sliced = df.copy()
    if start is not None:
        sliced = sliced.loc[sliced["date"] >= start]
    if end is not None:
        sliced = sliced.loc[sliced["date"] <= end]
    return sliced


def _comparison_row(
    strategy_name: str,
    weekly: pd.DataFrame,
    *,
    holding_period_days: int,
    top_n: int,
    use_regime_filter: bool,
    regime_exposure: float,
    analyst_count_threshold: int,
    min_avg_dollar_volume: float,
    analyst_data_is_point_in_time: bool,
) -> dict[str, dict]:
    analyst_data_mode = strategy_analyst_data_mode(strategy_name)
    periods = {
        "full": weekly,
        "dev": _slice_period(weekly, end=DEV_END),
        "test": _slice_period(weekly, start=TEST_START),
    }
    rows: dict[str, dict] = {}
    for label, period_df in periods.items():
        metrics = calculate_performance_metrics(period_df, holding_period_days=holding_period_days)
        rows[label] = {
            "strategy_name": strategy_name,
            "holding_period_days": holding_period_days,
            "top_n": top_n,
            "use_regime_filter": use_regime_filter,
            "regime_exposure": regime_exposure,
            "analyst_count_threshold": analyst_count_threshold,
            "min_avg_dollar_volume": min_avg_dollar_volume,
            "analyst_data_is_point_in_time": analyst_data_is_point_in_time,
            "analyst_data_mode": analyst_data_mode,
            **metrics,
        }
    return rows


def _save_comparisons(config: Config, period_rows: dict[str, list[dict]]) -> None:
    for label in ["full", "dev", "test"]:
        df = pd.DataFrame(period_rows[label]).sort_values("sharpe_ratio", ascending=False)
        save_dataframe(config.tables_dir / f"strategy_comparison_{label}.csv", df)
        if label == "full":
            save_dataframe(config.tables_dir / "strategy_comparison.csv", df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--holding-period-days", type=int, default=21)
    parser.add_argument("--analyst-count-threshold", type=int, default=10)
    parser.add_argument("--use-analyst-filters", default="true")
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--use-regime-filter", default="false")
    parser.add_argument("--regime-exposure", type=float, default=0.0)
    parser.add_argument("--min-avg-dollar-volume", type=float, default=20_000_000)
    parser.add_argument("--resistance-distance-threshold", type=float, default=0.02)
    parser.add_argument("--require-low-target-upside-4pct", default="false")
    parser.add_argument("--require-positive-revision-7d", default="false")
    parser.add_argument("--require-positive-revision-30d", default="false")
    parser.add_argument("--resistance-window", type=int, default=30)
    parser.add_argument("--max-names-per-sector", type=int, default=None)
    parser.add_argument("--use-inverse-vol-weighting", default="false")
    parser.add_argument("--enable-drawdown-protection", default="false")
    parser.add_argument("--run-condition-based", default="false")
    args = parser.parse_args()

    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    top_n = args.top_n if args.top_n is not None else config.top_n
    transaction_cost_bps = args.transaction_cost_bps if args.transaction_cost_bps is not None else config.transaction_cost_bps
    use_analyst_filters = str_to_bool(args.use_analyst_filters, default=True)
    use_regime_filter = str_to_bool(args.use_regime_filter, default=False)
    analyst_data_is_point_in_time = not (
        "analyst_data_mode" in features.columns and features["analyst_data_mode"].fillna("").eq("research_current_snapshot").any()
    )
    window_label = f"{features['date'].min().date()}_{features['date'].max().date()}"

    strategy_names = [
        "full_model",
        "strict_checklist_model",
        "technical_only",
        "technical_momentum_model",
        "analyst_snapshot_model",
    ]
    period_rows: dict[str, list[dict]] = {"full": [], "dev": [], "test": []}
    full_weekly: pd.DataFrame | None = None
    full_holdings: pd.DataFrame | None = None
    full_diagnostics: pd.DataFrame | None = None

    for strategy_name in strategy_names:
        weekly, holdings, diagnostics = run_weekly_backtest(
            features=features,
            holding_period_days=args.holding_period_days,
            benchmark=config.benchmark,
            top_n=top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            use_regime_filter=use_regime_filter,
            regime_exposure=args.regime_exposure,
            use_analyst_filters=use_analyst_filters,
            analyst_count_threshold=args.analyst_count_threshold,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            strategy_name=strategy_name,
            resistance_distance_threshold=args.resistance_distance_threshold,
            require_low_target_upside_4pct=str_to_bool(args.require_low_target_upside_4pct, default=False),
            require_positive_revision_7d=str_to_bool(args.require_positive_revision_7d, default=False),
            require_positive_revision_30d=str_to_bool(args.require_positive_revision_30d, default=False),
            resistance_window=args.resistance_window,
            max_names_per_sector=args.max_names_per_sector,
            use_inverse_vol_weighting=str_to_bool(args.use_inverse_vol_weighting, default=False),
            enable_drawdown_protection=str_to_bool(args.enable_drawdown_protection, default=False),
        )
        rows = _comparison_row(
            strategy_name,
            weekly,
            holding_period_days=args.holding_period_days,
            top_n=top_n,
            use_regime_filter=use_regime_filter,
            regime_exposure=args.regime_exposure,
            analyst_count_threshold=args.analyst_count_threshold,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            analyst_data_is_point_in_time=analyst_data_is_point_in_time,
        )
        for label in period_rows:
            period_rows[label].append(rows[label])

        prefix = f"{strategy_name}_hp{args.holding_period_days}"
        save_backtest_outputs(weekly, holdings, config.tables_dir, benchmark=config.benchmark, prefix=prefix)
        save_filter_diagnostics(diagnostics, config.tables_dir, prefix=prefix)
        if strategy_name == "full_model":
            full_weekly, full_holdings, full_diagnostics = weekly, holdings, diagnostics
            if rows["full"]["average_selected_count"] < 3:
                LOGGER.warning(
                    "full_model average selected count is very low (%.2f). Strategy may be under-diversified.",
                    rows["full"]["average_selected_count"],
                )

    _save_comparisons(config, period_rows)

    if full_weekly is not None and full_holdings is not None and full_diagnostics is not None:
        save_backtest_outputs(full_weekly, full_holdings, config.tables_dir, benchmark=config.benchmark)
        save_filter_diagnostics(full_diagnostics, config.tables_dir)
        save_dataframe(
            config.tables_dir / f"weekly_returns_h{args.holding_period_days}_{window_label}.csv",
            full_weekly,
        )
        save_dataframe(
            config.tables_dir / f"weekly_holdings_h{args.holding_period_days}_{window_label}.csv",
            full_holdings,
        )
        validation_df = save_backtest_validation(
            features=features,
            weekly_returns=full_weekly,
            output_dir=config.tables_dir,
            benchmark=config.benchmark,
            holding_period_days=args.holding_period_days,
        )
        if float(validation_df["compounded_spy_return_from_backtest"].iloc[0]) > 5:
            LOGGER.warning("Benchmark compounding appears unrealistic. Please inspect backtest_validation.csv.")
        create_plots(full_weekly, full_holdings, config.charts_dir)

    if str_to_bool(args.run_condition_based, default=False):
        cb_returns, cb_holdings, cb_trades = run_condition_based_backtest(
            features=features,
            strategy_name="strict_checklist_model",
            top_n=top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            benchmark=config.benchmark,
            analyst_count_threshold=args.analyst_count_threshold,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            use_regime_filter=use_regime_filter,
            regime_exposure=args.regime_exposure,
            use_inverse_vol_weighting=str_to_bool(args.use_inverse_vol_weighting, default=False),
            max_names_per_sector=args.max_names_per_sector,
            resistance_distance_threshold=args.resistance_distance_threshold,
            require_low_target_upside_4pct=str_to_bool(args.require_low_target_upside_4pct, default=False),
            require_positive_revision_7d=str_to_bool(args.require_positive_revision_7d, default=False),
            require_positive_revision_30d=str_to_bool(args.require_positive_revision_30d, default=False),
            resistance_window=args.resistance_window,
        )
        save_dataframe(config.tables_dir / "condition_based_returns.csv", cb_returns)
        save_dataframe(config.tables_dir / "condition_based_holdings.csv", cb_holdings)
        save_dataframe(config.tables_dir / "condition_based_trades.csv", cb_trades)


if __name__ == "__main__":
    main()
