from __future__ import annotations

import argparse
from itertools import product
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
WALK_FORWARD_WINDOWS = [
    ("2024 H1 OOS", "2024-01-01", "2024-06-30"),
    ("2024 H2 OOS", "2024-07-01", "2024-12-31"),
    ("2025 OOS", "2025-01-01", "2025-12-31"),
]


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
    for top_n, position_sizing, max_single_name_weight, max_names_per_sector, total_cost_bps in product(
        [5, 10, 15, 20],
        ["equal_weight", "inverse_volatility", "score_weighted", "score_over_volatility"],
        [0.15, 0.20],
        [None, 3, 4],
        [10, 20],
    ):
        weekly, _, _ = run_weekly_backtest(
            features=features,
            holding_period_days=5,
            benchmark=config.benchmark,
            top_n=top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=total_cost_bps,
            use_regime_filter=False,
            regime_exposure=0.0,
            use_analyst_filters=False,
            analyst_count_threshold=config.analyst_count_threshold,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            strategy_name="final_quant_5d_no_snapshot_no_sma_filter",
            max_names_per_sector=max_names_per_sector,
            max_single_name_weight=max_single_name_weight,
            use_inverse_vol_weighting=position_sizing == "inverse_volatility",
            position_sizing=position_sizing,
            min_historical_rating_count=5,
        )
        walk_rows = []
        for window_label, start, end in WALK_FORWARD_WINDOWS:
            metrics = _safe_metrics(_slice_period(weekly, start, end), 5)
            walk_rows.append(
                {
                    "window_label": window_label,
                    "excess": metrics["excess_total_return"],
                    "beat_spy": bool(pd.notna(metrics["excess_total_return"]) and metrics["excess_total_return"] > 0),
                }
            )
        walk_df = pd.DataFrame(walk_rows)
        test_metrics = _safe_metrics(_slice_period(weekly, "2025-01-01", "2025-12-31"), 5)
        full_metrics = _safe_metrics(weekly, 5)
        rows.append(
            {
                "strategy_name": "final_quant_5d_no_snapshot_no_sma_filter",
                "display_name": strategy_display_name("final_quant_5d_no_snapshot_no_sma_filter"),
                "holding_period_days": 5,
                "top_n": top_n,
                "position_sizing": position_sizing,
                "max_single_name_weight": max_single_name_weight,
                "max_names_per_sector": max_names_per_sector,
                "total_cost_bps": total_cost_bps,
                "regime_filter": "none",
                "long_short": False,
                "walk_forward_average_excess_vs_spy": float(walk_df["excess"].mean()),
                "windows_beating_spy": int(walk_df["beat_spy"].sum()),
                "test_2025_excess_vs_spy": test_metrics["excess_total_return"],
                "max_drawdown": full_metrics["max_drawdown"],
                "average_turnover": full_metrics["average_turnover"],
                "average_holdings": full_metrics["average_selected_count"],
                "is_baseline": bool(
                    top_n == 10
                    and position_sizing == "equal_weight"
                    and max_single_name_weight == 0.15
                    and max_names_per_sector is None
                    and total_cost_bps == 10
                ),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["walk_forward_average_excess_vs_spy", "windows_beating_spy", "test_2025_excess_vs_spy", "max_drawdown", "average_turnover"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    results_df["selected_best"] = False
    if not results_df.empty:
        results_df.loc[0, "selected_best"] = True

    save_dataframe(config.tables_dir / "position_sizing_comparison.csv", results_df)
    baseline_row = results_df.loc[results_df["is_baseline"]].iloc[0]
    report_lines = [
        "# Position Sizing Comparison",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        f"- Baseline row: {baseline_row['display_name']}, top_n=10, equal_weight, max_single_name_weight=0.15, sector_cap=None, total_cost_bps=10, regime_filter=none, long_short=false",
        f"- Best sizing configuration: {results_df.iloc[0]['position_sizing']} / top_n={int(results_df.iloc[0]['top_n'])} / sector_cap={results_df.iloc[0]['max_names_per_sector']} / cost_bps={int(results_df.iloc[0]['total_cost_bps'])}",
        "",
        "## Results",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    (config.reports_dir / "position_sizing_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved position sizing table to {config.tables_dir / 'position_sizing_comparison.csv'}")
    print(f"Saved position sizing report to {config.reports_dir / 'position_sizing_comparison.md'}")


if __name__ == "__main__":
    main()
