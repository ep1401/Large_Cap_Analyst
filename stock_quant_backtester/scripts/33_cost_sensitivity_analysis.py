from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import strategy_display_name
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical."
SNAPSHOT_CAVEAT = "Snapshot analyst target models are excluded from the main historically safer ranking."
HISTORICAL_NOTE = "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date."
SENTIMENT_CAVEAT = "News sentiment depends on Alpha Vantage coverage and classification."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
REGIME_CAVEAT = "Regime filters were tested and are not recommended for the main model based on current results."
RESEARCH_CAVEAT = "This is research/paper trading only, not financial advice."
TEST_START = pd.Timestamp("2025-01-01")
STRATEGIES = [
    "final_quant_5d_no_snapshot_no_sma_filter",
    "historical_rating_counts_plus_events",
    "final_quant_5d_no_snapshot",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])

    rows: list[dict] = []
    for strategy_name in STRATEGIES:
        for total_cost_bps in [0, 5, 10, 20, 30, 50]:
            weekly, _, _ = run_weekly_backtest(
                features=features,
                holding_period_days=5,
                benchmark=config.benchmark,
                top_n=10,
                initial_capital=config.initial_capital,
                transaction_cost_bps=total_cost_bps,
                use_regime_filter=False,
                regime_exposure=0.0,
                use_analyst_filters=False,
                analyst_count_threshold=config.analyst_count_threshold,
                min_avg_dollar_volume=config.min_avg_dollar_volume,
                strategy_name=strategy_name,
                position_sizing="equal_weight",
                min_historical_rating_count=5,
            )
            full = _safe_metrics(weekly, 5)
            test = _safe_metrics(_slice_period(weekly, TEST_START), 5)
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "display_name": strategy_display_name(strategy_name),
                    "holding_period_days": 5,
                    "top_n": 10,
                    "position_sizing": "equal_weight",
                    "regime_filter": "none",
                    "long_short": False,
                    "total_cost_bps": total_cost_bps,
                    "commission_bps": 0.0,
                    "spread_bps": total_cost_bps,
                    "slippage_bps": 0.0,
                    "market_impact_bps": 0.0,
                    "test_excess_vs_spy": test["excess_total_return"],
                    "test_sharpe": test["sharpe_ratio"],
                    "max_drawdown": full["max_drawdown"],
                    "average_turnover": full["average_turnover"],
                    "is_baseline": bool(strategy_name == "final_quant_5d_no_snapshot_no_sma_filter" and total_cost_bps == 10),
                }
            )

    results_df = pd.DataFrame(rows).sort_values(["strategy_name", "total_cost_bps"]).reset_index(drop=True)
    break_even_rows = []
    for strategy_name, group in results_df.groupby("strategy_name"):
        beating = group.loc[group["test_excess_vs_spy"] > 0, "total_cost_bps"]
        break_even = float(beating.max()) if not beating.empty else float("nan")
        break_even_rows.append({"strategy_name": strategy_name, "break_even_cost_bps": break_even})
    break_even_df = pd.DataFrame(break_even_rows)
    results_df = results_df.merge(break_even_df, on="strategy_name", how="left")

    save_dataframe(config.tables_dir / "cost_sensitivity_analysis.csv", results_df)
    baseline_row = results_df.loc[results_df["is_baseline"]].iloc[0]
    report_lines = [
        "# Cost Sensitivity Analysis",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        f"- Baseline row: {baseline_row['display_name']}, top_n=10, equal_weight, total_cost_bps=10, regime_filter=none, long_short=false",
        f"- Baseline test excess vs SPY: {float(baseline_row['test_excess_vs_spy']):.2%}",
        f"- Baseline average turnover: {float(baseline_row['average_turnover']):.4f}",
        f"- Baseline max drawdown: {float(baseline_row['max_drawdown']):.2%}",
        f"- Current 10 bps assumption appears conservative enough: {bool(float(baseline_row['break_even_cost_bps']) >= 10) if pd.notna(baseline_row['break_even_cost_bps']) else False}",
        "",
        "## Results",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    (config.reports_dir / "cost_sensitivity_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved cost sensitivity table to {config.tables_dir / 'cost_sensitivity_analysis.csv'}")
    print(f"Saved cost sensitivity report to {config.reports_dir / 'cost_sensitivity_analysis.md'}")


if __name__ == "__main__":
    main()
