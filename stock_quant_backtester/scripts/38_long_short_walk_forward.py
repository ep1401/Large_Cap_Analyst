from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_long_short_backtest, run_weekly_backtest
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import strategy_display_name
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Backtested long/short returns are hypothetical."
SHORT_RISK_CAVEAT = "Shorting can create unlimited losses."
BORROW_CAVEAT = "Borrow costs and stock-loan availability are simplified assumptions."
RESEARCH_CAVEAT = "This is research and paper-trading only, not financial advice."
LONG_SHORT_NOT_RECOMMENDED = (
    "Long/short variants were tested but are not recommended because the short book had negative average contribution and reduced walk-forward robustness."
)
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
STRATEGIES = {
    "final_quant_5d_no_snapshot_no_sma_filter": {
        "holding_period_days": 5,
        "kind": "long_only",
        "peer": None,
    },
    "long_short_5d_no_snapshot_100_50": {
        "holding_period_days": 5,
        "kind": "long_short",
        "peer": "final_quant_5d_no_snapshot_no_sma_filter",
        "long_exposure": 1.0,
        "short_exposure": 0.5,
    },
    "long_short_5d_no_snapshot_100_100": {
        "holding_period_days": 5,
        "kind": "long_short",
        "peer": "final_quant_5d_no_snapshot_no_sma_filter",
        "long_exposure": 1.0,
        "short_exposure": 1.0,
    },
    "final_quant_21d_no_snapshot_sector_capped": {
        "holding_period_days": 21,
        "kind": "long_only",
        "peer": None,
    },
    "long_short_21d_no_snapshot_100_50": {
        "holding_period_days": 21,
        "kind": "long_short",
        "peer": "final_quant_21d_no_snapshot_sector_capped",
        "long_exposure": 1.0,
        "short_exposure": 0.5,
    },
    "long_short_21d_no_snapshot_100_100": {
        "holding_period_days": 21,
        "kind": "long_short",
        "peer": "final_quant_21d_no_snapshot_sector_capped",
        "long_exposure": 1.0,
        "short_exposure": 1.0,
    },
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])

    strategy_runs: dict[str, pd.DataFrame] = {}
    for strategy_name, spec in STRATEGIES.items():
        if spec["kind"] == "long_only":
            weekly, _, _ = run_weekly_backtest(
                features=features,
                holding_period_days=spec["holding_period_days"],
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
        else:
            weekly, _, _, _ = run_long_short_backtest(
                features=features,
                strategy_name=strategy_name,
                holding_period_days=spec["holding_period_days"],
                long_n=config.top_n,
                short_n=10,
                long_exposure=spec["long_exposure"],
                short_exposure=spec["short_exposure"],
                benchmark=config.benchmark,
                transaction_cost_bps=config.transaction_cost_bps,
                short_borrow_bps_annual=300,
                extra_short_slippage_bps=5,
                max_single_name_weight=0.15,
                min_avg_dollar_volume=config.min_avg_dollar_volume,
            )
        strategy_runs[strategy_name] = weekly

    rows: list[dict] = []
    for strategy_name, spec in STRATEGIES.items():
        weekly = strategy_runs[strategy_name]
        for window in WALK_FORWARD_WINDOWS:
            sliced = _slice_period(weekly, window["test_start"], window["test_end"])
            metrics = _safe_metrics(sliced, spec["holding_period_days"])
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "display_name": strategy_display_name(strategy_name),
                    "holding_period_days": spec["holding_period_days"],
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
                    "average_gross_exposure": float(sliced["gross_exposure"].mean()) if "gross_exposure" in sliced.columns and not sliced.empty else 1.0,
                    "average_net_exposure": float(sliced["net_exposure"].mean()) if "net_exposure" in sliced.columns and not sliced.empty else 1.0,
                    "short_contribution": float(sliced["short_contribution"].sum()) if "short_contribution" in sliced.columns and not sliced.empty else 0.0,
                    "percent_periods_short_helped": float(sliced["short_book_helped"].mean()) if "short_book_helped" in sliced.columns and not sliced.empty else 0.0,
                    "beat_spy": bool(metrics["excess_total_return"] > 0) if pd.notna(metrics["excess_total_return"]) else False,
                }
            )

    results_df = pd.DataFrame(rows).sort_values(["holding_period_days", "strategy_name", "test_start"])

    summary_rows: list[dict] = []
    for strategy_name, spec in STRATEGIES.items():
        strategy_df = results_df.loc[results_df["strategy_name"] == strategy_name].copy()
        summary = {
            "strategy_name": strategy_name,
            "display_name": strategy_display_name(strategy_name),
            "holding_period_days": spec["holding_period_days"],
            "windows_beating_spy": int(strategy_df["beat_spy"].sum()),
            "average_excess_return": float(strategy_df["test_excess_return_vs_spy"].mean()),
            "worst_window": strategy_df.sort_values("test_excess_return_vs_spy").iloc[0]["window_label"],
            "worst_window_excess_return": float(strategy_df["test_excess_return_vs_spy"].min()),
            "average_drawdown": float(strategy_df["max_drawdown"].mean()),
            "average_short_contribution": float(strategy_df["short_contribution"].mean()),
            "average_percent_periods_short_helped": float(strategy_df["percent_periods_short_helped"].mean()),
            "robustness_vs_long_only": "n/a",
        }
        if spec["kind"] == "long_short":
            peer_df = results_df.loc[results_df["strategy_name"] == spec["peer"]].copy()
            delta_beats = int(strategy_df["beat_spy"].sum()) - int(peer_df["beat_spy"].sum())
            delta_excess = float(strategy_df["test_excess_return_vs_spy"].mean() - peer_df["test_excess_return_vs_spy"].mean())
            if delta_beats > 0 and delta_excess > 0:
                robustness = "improves"
            elif delta_beats < 0 or delta_excess < 0:
                robustness = "hurts"
            else:
                robustness = "mixed"
            summary["robustness_vs_long_only"] = robustness
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["holding_period_days", "average_excess_return", "windows_beating_spy"],
        ascending=[True, False, False],
    )
    save_dataframe(config.tables_dir / "long_short_walk_forward_results.csv", results_df)

    report_lines = [
        "# Long Short Walk Forward Summary",
        "",
        f"- {LONG_SHORT_NOT_RECOMMENDED}",
        f"- {SHORT_RISK_CAVEAT}",
        f"- {BORROW_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(summary_df.round(6)),
        "",
        "## Window Detail",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    (config.reports_dir / "long_short_walk_forward_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved long/short walk-forward results to {config.tables_dir / 'long_short_walk_forward_results.csv'}")
    print(f"Saved long/short walk-forward summary to {config.reports_dir / 'long_short_walk_forward_summary.md'}")


if __name__ == "__main__":
    main()
