from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, save_backtest_outputs
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.plots import create_plots
from src.utils import LOGGER, load_dataframe, save_dataframe, str_to_bool


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
    args = parser.parse_args()

    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    top_n = args.top_n if args.top_n is not None else config.top_n
    transaction_cost_bps = (
        args.transaction_cost_bps if args.transaction_cost_bps is not None else config.transaction_cost_bps
    )
    use_analyst_filters = str_to_bool(args.use_analyst_filters, default=True)
    use_regime_filter = str_to_bool(args.use_regime_filter, default=False)

    strategy_names = ["full_model", "technical_only", "analyst_only"]
    comparison_rows: list[dict] = []
    full_weekly: pd.DataFrame | None = None
    full_holdings: pd.DataFrame | None = None

    for strategy_name in strategy_names:
        weekly, holdings = run_weekly_backtest(
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
        )
        metrics = calculate_performance_metrics(weekly)
        comparison_rows.append(
            {
                "strategy_name": strategy_name,
                "holding_period_days": args.holding_period_days,
                "top_n": top_n,
                "use_regime_filter": use_regime_filter,
                "regime_exposure": args.regime_exposure,
                "analyst_count_threshold": args.analyst_count_threshold,
                "min_avg_dollar_volume": args.min_avg_dollar_volume,
                **metrics,
            }
        )
        save_dataframe(config.tables_dir / f"{strategy_name}_weekly_portfolio_returns.csv", weekly)
        save_dataframe(config.tables_dir / f"{strategy_name}_weekly_holdings.csv", holdings)
        if strategy_name == "full_model":
            full_weekly, full_holdings = weekly, holdings
            if metrics["average_selected_count"] < 3:
                LOGGER.warning("full_model average selected count is very low (%.2f). Strategy may be under-diversified.", metrics["average_selected_count"])

    comparison = pd.DataFrame(comparison_rows).sort_values("sharpe_ratio", ascending=False)
    save_dataframe(config.tables_dir / "strategy_comparison.csv", comparison)

    if full_weekly is not None and full_holdings is not None:
        save_backtest_outputs(full_weekly, full_holdings, config.tables_dir, benchmark=config.benchmark)
        create_plots(full_weekly, full_holdings, config.charts_dir)


if __name__ == "__main__":
    main()
