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


BACKTEST_CAVEAT = "Back-tested performance is hypothetical and does not reflect actual live performance."
SNAPSHOT_CAVEAT = (
    "Important caveat: snapshot analyst target models are excluded from the main historically safer ranking because current target data is not point-in-time historical data."
)
HISTORICAL_NOTE = (
    "Historical rating-count features are built from dated FMP grades-historical records and use only the latest record available on or before each rebalance date."
)
SENTIMENT_CAVEAT = (
    "News sentiment results depend on Alpha Vantage coverage, ticker relevance scoring, publication timestamps, and provider sentiment classification."
)
RESEARCH_CAVEAT = "This is a historical research backtest, not financial advice and not a live trading system."

WALK_FORWARD_WINDOWS = [
    {
        "window_label": "2024 H1 OOS",
        "train_start": "2023-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
    },
    {
        "window_label": "2024 H2 OOS",
        "train_start": "2023-01-01",
        "train_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
    {
        "window_label": "2025 OOS",
        "train_start": "2023-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
]
HORIZON_STRATEGIES = {
    5: [
        "final_quant_5d_no_snapshot",
        "final_quant_5d_no_snapshot_loose",
        "final_quant_5d_no_snapshot_no_sma_filter",
    ],
    21: [
        "final_quant_21d_no_snapshot",
        "final_quant_21d_no_snapshot_with_sma_filter",
        "final_quant_21d_no_snapshot_sector_capped",
    ],
    63: [
        "final_quant_63d_no_snapshot",
        "final_quant_63d_no_snapshot_with_sma200_filter",
        "final_quant_63d_no_snapshot_sector_capped",
    ],
}


def _slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()


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


def _stability_label(summary_row: pd.Series) -> str:
    windows = int(summary_row["number_of_out_of_sample_windows"])
    beating = int(summary_row["windows_beating_spy"])
    avg_excess = float(summary_row["average_excess_return_vs_spy"])
    worst_excess = float(summary_row["worst_out_of_sample_excess_return"])
    if windows == 0:
        return "insufficient_data"
    if beating >= max(2, (windows + 1) // 2) and avg_excess > 0 and worst_excess > -0.10:
        return "stable"
    if beating == 0 or avg_excess <= 0:
        return "weak"
    return "concentrated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])

    result_rows: list[dict] = []
    summary_rows: list[dict] = []

    for holding_period_days, strategies in HORIZON_STRATEGIES.items():
        for strategy_name in strategies:
            max_names_per_sector = 3 if strategy_name.endswith("sector_capped") else None
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
                max_names_per_sector=max_names_per_sector,
                min_historical_rating_count=5,
            )

            strategy_window_rows: list[dict] = []
            for window in WALK_FORWARD_WINDOWS:
                sliced = _slice_period(weekly, window["test_start"], window["test_end"])
                metrics = _safe_metrics(sliced, holding_period_days)
                row = {
                    "holding_period_days": holding_period_days,
                    "strategy_name": strategy_name,
                    "display_name": strategy_display_name(strategy_name),
                    "window_label": window["window_label"],
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "test_total_return": metrics["total_return"],
                    "test_excess_return_vs_spy": metrics["excess_total_return"],
                    "test_sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "average_turnover": metrics["average_turnover"],
                    "average_holdings": metrics["average_selected_count"],
                    "number_of_rebalance_periods": metrics["number_of_rebalance_periods"],
                    "periods_beating_spy": metrics["weeks_beating_spy"],
                    "beat_spy": bool(metrics["excess_total_return"] > 0) if pd.notna(metrics["excess_total_return"]) else False,
                }
                result_rows.append(row)
                strategy_window_rows.append(row)

            strategy_window_df = pd.DataFrame(strategy_window_rows)
            summary = {
                "holding_period_days": holding_period_days,
                "strategy_name": strategy_name,
                "display_name": strategy_display_name(strategy_name),
                "number_of_out_of_sample_windows": int(strategy_window_df["window_label"].nunique()),
                "windows_beating_spy": int(strategy_window_df["beat_spy"].sum()),
                "average_excess_return_vs_spy": float(strategy_window_df["test_excess_return_vs_spy"].mean()),
                "worst_out_of_sample_excess_return": float(strategy_window_df["test_excess_return_vs_spy"].min()),
                "average_max_drawdown": float(strategy_window_df["max_drawdown"].mean()),
            }
            summary["performance_profile"] = _stability_label(pd.Series(summary))
            summary_rows.append(summary)

    results_df = pd.DataFrame(result_rows).sort_values(
        ["holding_period_days", "strategy_name", "test_start"],
        ascending=[True, True, True],
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["holding_period_days", "average_excess_return_vs_spy", "windows_beating_spy", "average_max_drawdown"],
        ascending=[True, False, False, False],
    )

    save_dataframe(config.tables_dir / "horizon_specific_walk_forward_results.csv", results_df)

    lines = [
        "# Horizon Specific Walk Forward Summary",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(summary_df.round(6)),
    ]

    for holding_period_days in sorted(HORIZON_STRATEGIES):
        horizon_window_df = results_df.loc[results_df["holding_period_days"] == holding_period_days].copy()
        lines.extend(
            [
                "",
                f"## {holding_period_days}-Day Out-of-Sample Windows",
                "",
                _dataframe_to_markdown(horizon_window_df.round(6)),
            ]
        )

    (config.reports_dir / "horizon_specific_walk_forward_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved walk-forward results to {config.tables_dir / 'horizon_specific_walk_forward_results.csv'}")
    print(f"Saved walk-forward summary to {config.reports_dir / 'horizon_specific_walk_forward_summary.md'}")


if __name__ == "__main__":
    main()
