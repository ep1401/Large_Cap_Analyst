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
    "Important caveat: analyst-driven results currently use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)


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

    strategy_names = ["full_model", "analyst_only", "technical_only"]
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
        weekly, _ = run_weekly_backtest(
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
        )
        metrics = calculate_performance_metrics(weekly)
        results.append(
            {
                "strategy_name": strategy_name,
                "top_n": top_n,
                "holding_period_days": holding_period_days,
                "analyst_count_threshold": effective_threshold,
                "use_regime_filter": use_regime_filter,
                "regime_exposure": regime_exposure,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["sharpe_ratio", "excess_total_return", "max_drawdown"],
        ascending=[False, False, False],
    )
    save_dataframe(config.tables_dir / "grid_search_results.csv", results_df)

    top_sharpe = results_df.head(10).round(6)
    top_excess = results_df.sort_values("excess_total_return", ascending=False).head(10).round(6)
    best_by_strategy = (
        results_df.sort_values(by=["strategy_name", "sharpe_ratio", "excess_total_return"], ascending=[True, False, False])
        .groupby("strategy_name", as_index=False)
        .head(1)
        .round(6)
    )

    lines = [
        "# Grid Search Summary",
        "",
        f"- Evaluated configurations: {len(results_df)}",
        f"- {IMPORTANT_CAVEAT}",
        "",
        "## Top 10 By Sharpe",
        "",
        _dataframe_to_markdown(top_sharpe),
        "",
        "## Top 10 By Excess Return Vs SPY",
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
