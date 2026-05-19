from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_long_short_backtest, run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import get_future_return_columns, strategy_display_name
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Backtested long/short returns are hypothetical."
SHORT_RISK_CAVEAT = "Shorting can create unlimited losses."
BORROW_CAVEAT = "Borrow costs and stock-loan availability are simplified assumptions."
RESEARCH_CAVEAT = "This is research and paper-trading only, not financial advice."
LONG_SHORT_NOT_RECOMMENDED = (
    "Long/short variants were tested but are not recommended because the short book had negative average contribution and reduced walk-forward robustness."
)
TEST_START = pd.Timestamp("2025-01-01")
LONG_ONLY_STRATEGIES = {
    5: "final_quant_5d_no_snapshot_no_sma_filter",
    21: "final_quant_21d_no_snapshot_sector_capped",
}
LONG_SHORT_VARIANTS = {
    "long_short_5d_no_snapshot_100_50": {"holding_period_days": 5, "long_exposure": 1.0, "short_exposure": 0.5},
    "long_short_5d_no_snapshot_100_100": {"holding_period_days": 5, "long_exposure": 1.0, "short_exposure": 1.0},
    "long_short_21d_no_snapshot_100_50": {"holding_period_days": 21, "long_exposure": 1.0, "short_exposure": 0.5},
    "long_short_21d_no_snapshot_100_100": {"holding_period_days": 21, "long_exposure": 1.0, "short_exposure": 1.0},
}


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out["date"] >= start]
    return out


def _safe_metrics(frame: pd.DataFrame, holding_period_days: int) -> dict[str, float]:
    if frame.empty:
        return {
            "total_return": float("nan"),
            "excess_total_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "average_turnover": float("nan"),
            "average_selected_count": float("nan"),
            "number_of_rebalance_periods": 0,
            "weeks_beating_spy": float("nan"),
        }
    return calculate_performance_metrics(frame, holding_period_days=holding_period_days)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def _build_spy_weekly(features: pd.DataFrame, config: Config, holding_period_days: int) -> pd.DataFrame:
    _, future_spy_return_column, _ = get_future_return_columns(holding_period_days)
    benchmark_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=config.benchmark)
    benchmark_rows = (
        features.loc[
            (features["ticker"] == config.benchmark)
            & (features["date"].isin(benchmark_dates))
            & features[future_spy_return_column].notna(),
            ["date", future_spy_return_column],
        ]
        .drop_duplicates("date")
        .sort_values("date")
    )
    portfolio_value = config.initial_capital
    spy_value = config.initial_capital
    rows: list[dict] = []
    for row in benchmark_rows.itertuples(index=False):
        ret = float(getattr(row, future_spy_return_column))
        portfolio_value *= 1 + ret
        spy_value *= 1 + ret
        rows.append(
            {
                "date": pd.to_datetime(row.date),
                "strategy_name": "SPY",
                "selected_count": 1,
                "gross_return": ret,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "extra_short_slippage": 0.0,
                "borrow_cost": 0.0,
                "net_return": ret,
                "spy_return": ret,
                "excess_return": 0.0,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "gross_exposure": 1.0,
                "net_exposure": 1.0,
                "long_exposure": 1.0,
                "short_exposure": 0.0,
                "long_contribution": ret,
                "short_contribution": 0.0,
                "average_long_return": ret,
                "average_short_book_return": 0.0,
                "short_book_helped": False,
                "short_book_hurt": False,
            }
        )
    return pd.DataFrame(rows)


def _summary_row(
    strategy_name: str,
    weekly: pd.DataFrame,
    holding_period_days: int,
    short_n: int | None,
    borrow_bps: float | None,
    extra_short_slippage_bps: float | None,
) -> dict[str, float | int | str | None]:
    full = _safe_metrics(weekly, holding_period_days)
    test = _safe_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
    return {
        "holding_period_days": holding_period_days,
        "strategy_name": strategy_name,
        "display_name": strategy_display_name(strategy_name),
        "short_n": short_n,
        "short_borrow_bps_annual": borrow_bps,
        "extra_short_slippage_bps": extra_short_slippage_bps,
        "full_period_return": full["total_return"],
        "test_period_return": test["total_return"],
        "test_period_excess_vs_spy": test["excess_total_return"],
        "sharpe": full["sharpe_ratio"],
        "max_drawdown": full["max_drawdown"],
        "gross_exposure": float(weekly["gross_exposure"].mean()) if "gross_exposure" in weekly.columns and not weekly.empty else 1.0,
        "net_exposure": float(weekly["net_exposure"].mean()) if "net_exposure" in weekly.columns and not weekly.empty else 1.0,
        "average_long_return": float(weekly["average_long_return"].mean()) if "average_long_return" in weekly.columns and not weekly.empty else float("nan"),
        "average_short_book_return": float(weekly["average_short_book_return"].mean()) if "average_short_book_return" in weekly.columns and not weekly.empty else 0.0,
        "short_contribution": float(weekly["short_contribution"].sum()) if "short_contribution" in weekly.columns and not weekly.empty else 0.0,
        "borrow_cost": float(weekly["borrow_cost"].sum()) if "borrow_cost" in weekly.columns and not weekly.empty else 0.0,
        "trading_cost": float((weekly["transaction_cost"] + weekly.get("extra_short_slippage", 0.0)).sum()) if not weekly.empty else 0.0,
        "turnover": float(weekly["turnover"].mean()) if "turnover" in weekly.columns and not weekly.empty else 0.0,
        "percent_periods_short_helped": float(weekly["short_book_helped"].mean()) if "short_book_helped" in weekly.columns and not weekly.empty else 0.0,
        "percent_periods_short_hurt": float(weekly["short_book_hurt"].mean()) if "short_book_hurt" in weekly.columns and not weekly.empty else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])

    rows: list[dict] = []

    for holding_period_days, strategy_name in LONG_ONLY_STRATEGIES.items():
        weekly, _, _ = run_weekly_backtest(
            features=features,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            top_n=config.top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            use_regime_filter=False,
            regime_exposure=0.0,
            use_analyst_filters=False,
            analyst_count_threshold=config.analyst_count_threshold,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            strategy_name=strategy_name,
            max_names_per_sector=3 if strategy_name.endswith("sector_capped") else None,
            min_historical_rating_count=5,
        )
        rows.append(_summary_row(strategy_name, weekly, holding_period_days, None, None, None))

        spy_weekly = _build_spy_weekly(features, config, holding_period_days)
        rows.append(_summary_row("SPY", spy_weekly, holding_period_days, None, None, None))

    for strategy_name, spec in LONG_SHORT_VARIANTS.items():
        for short_n in [5, 10, 15]:
            for borrow_bps in [100, 300, 500]:
                for extra_slippage_bps in [5, 10, 20]:
                    weekly, _, _, _ = run_long_short_backtest(
                        features=features,
                        strategy_name=strategy_name,
                        holding_period_days=spec["holding_period_days"],
                        long_n=config.top_n,
                        short_n=short_n,
                        long_exposure=spec["long_exposure"],
                        short_exposure=spec["short_exposure"],
                        benchmark=config.benchmark,
                        transaction_cost_bps=config.transaction_cost_bps,
                        short_borrow_bps_annual=borrow_bps,
                        extra_short_slippage_bps=extra_slippage_bps,
                        max_single_name_weight=0.15,
                        min_avg_dollar_volume=config.min_avg_dollar_volume,
                    )
                    rows.append(
                        _summary_row(
                            strategy_name,
                            weekly,
                            spec["holding_period_days"],
                            short_n,
                            borrow_bps,
                            extra_slippage_bps,
                        )
                    )

    comparison_df = pd.DataFrame(rows).sort_values(
        ["holding_period_days", "strategy_name", "short_n", "short_borrow_bps_annual", "extra_short_slippage_bps"],
        na_position="first",
    )
    save_dataframe(config.tables_dir / "long_short_comparison.csv", comparison_df)

    report_lines = [
        "# Long Short Comparison",
        "",
        f"- {LONG_SHORT_NOT_RECOMMENDED}",
        f"- {SHORT_RISK_CAVEAT}",
        f"- {BORROW_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(comparison_df.round(6)),
    ]
    (config.reports_dir / "long_short_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved long/short comparison to {config.tables_dir / 'long_short_comparison.csv'}")
    print(f"Saved long/short comparison report to {config.reports_dir / 'long_short_comparison.md'}")


if __name__ == "__main__":
    main()
