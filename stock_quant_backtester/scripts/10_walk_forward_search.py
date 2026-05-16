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
    out = df.copy()
    if start is not None:
        out = out.loc[out["date"] >= start]
    if end is not None:
        out = out.loc[out["date"] <= end]
    return out


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

    strategy_names = ["full_model", "strict_checklist_model", "technical_momentum_model"]
    holding_period_days_values = [5, 21, 63]
    top_ns = [5, 10, 15, 20]
    analyst_thresholds = [10, 20, 30]
    resistance_thresholds = [0.02, 0.03, 0.05]
    use_regime_filters = [False, True]
    max_names_options = [None, 3]
    inverse_vol_options = [False, True]

    rows: list[dict] = []
    partial_results_path = config.tables_dir / "walk_forward_search_results_partial.csv"
    combinations: list[dict] = []
    for strategy_name in strategy_names:
        strategy_analyst_thresholds = analyst_thresholds if strategy_name in {"full_model", "strict_checklist_model"} else [10]
        strategy_resistance_thresholds = resistance_thresholds if strategy_name == "strict_checklist_model" else [0.03]
        for holding_period_days, top_n, analyst_threshold, resistance_threshold, use_regime_filter, max_names_per_sector, use_inverse_vol_weighting in product(
            holding_period_days_values,
            top_ns,
            strategy_analyst_thresholds,
            strategy_resistance_thresholds,
            use_regime_filters,
            max_names_options,
            inverse_vol_options,
        ):
            combinations.append(
                {
                    "strategy_name": strategy_name,
                    "holding_period_days": holding_period_days,
                    "top_n": top_n,
                    "analyst_threshold": analyst_threshold,
                    "resistance_threshold": resistance_threshold,
                    "use_regime_filter": use_regime_filter,
                    "max_names_per_sector": max_names_per_sector,
                    "use_inverse_vol_weighting": use_inverse_vol_weighting,
                }
            )

    total_combinations = len(combinations)
    for index, combo in enumerate(combinations, start=1):
        strategy_name = combo["strategy_name"]
        holding_period_days = combo["holding_period_days"]
        top_n = combo["top_n"]
        analyst_threshold = combo["analyst_threshold"]
        resistance_threshold = combo["resistance_threshold"]
        use_regime_filter = combo["use_regime_filter"]
        max_names_per_sector = combo["max_names_per_sector"]
        use_inverse_vol_weighting = combo["use_inverse_vol_weighting"]
        if index == 1 or index % 25 == 0 or index == total_combinations:
            print(f"Running walk-forward combo {index}/{total_combinations}: {strategy_name}, hp={holding_period_days}, top_n={top_n}")
        weekly, _, _ = run_weekly_backtest(
            features=features,
            strategy_name=strategy_name,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            top_n=top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            use_regime_filter=use_regime_filter,
            regime_exposure=0.0,
            analyst_count_threshold=analyst_threshold,
            min_avg_dollar_volume=20_000_000,
            resistance_distance_threshold=resistance_threshold,
            max_names_per_sector=max_names_per_sector,
            use_inverse_vol_weighting=use_inverse_vol_weighting,
            require_positive_revision_7d=False,
            resistance_window=63 if strategy_name == "strict_checklist_model" else 30,
        )
        dev_metrics = calculate_performance_metrics(_slice_period(weekly, end=DEV_END), holding_period_days)
        test_metrics = calculate_performance_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
        rows.append(
            {
                "strategy_name": strategy_name,
                "holding_period_days": holding_period_days,
                "top_n": top_n,
                "analyst_count_threshold": analyst_threshold,
                "resistance_distance_threshold": resistance_threshold,
                "use_regime_filter": use_regime_filter,
                "max_names_per_sector": max_names_per_sector,
                "use_inverse_vol_weighting": use_inverse_vol_weighting,
                **{f"development_{k}": v for k, v in dev_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        if index % 25 == 0 or index == total_combinations:
            save_dataframe(partial_results_path, pd.DataFrame(rows))

    results = pd.DataFrame(rows)
    preferred = results.loc[results["development_max_drawdown"] > -0.30]
    search_base = preferred if not preferred.empty else results
    best_configs = (
        search_base.sort_values(
            by=["development_sharpe_ratio", "development_excess_total_return", "development_max_drawdown"],
            ascending=[False, False, False],
        )
        .groupby("strategy_name", as_index=False)
        .head(3)
    )

    ordered = results.sort_values(
        by=["test_sharpe_ratio", "test_excess_total_return", "test_max_drawdown"],
        ascending=[False, False, False],
    )
    save_dataframe(config.tables_dir / "walk_forward_search_results.csv", ordered)

    lines = [
        "# Walk Forward Search Summary",
        "",
        f"- Evaluated configurations: {len(results)}",
        f"- {IMPORTANT_CAVEAT}",
        "",
        "## Best Development Configurations",
        "",
        _dataframe_to_markdown(best_configs.round(6)),
        "",
        "## Top Test Results",
        "",
        _dataframe_to_markdown(ordered.head(10).round(6)),
        "",
        "## Overfit Warning",
        "- Results should be judged primarily on test-period performance. A strategy should not be considered stronger just because it wins on the full period.",
    ]
    report_path = config.reports_dir / "walk_forward_search_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved walk-forward results to {config.tables_dir / 'walk_forward_search_results.csv'}")
    print(f"Saved walk-forward summary to {report_path}")


if __name__ == "__main__":
    main()
