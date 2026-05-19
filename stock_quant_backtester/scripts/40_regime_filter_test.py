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

    best_strategy = "final_quant_5d_no_snapshot_no_sma_filter"
    base_top_n = config.top_n
    base_sector_cap = None
    base_position_sizing = "equal_weight"
    base_cost_bps = config.transaction_cost_bps
    base_avoid_negative_news = False
    base_avoid_recent_downgrades = False
    optimization_path = config.tables_dir / "final_5d_long_only_optimization.csv"
    if optimization_path.exists():
        optimization_df = load_dataframe(optimization_path)
        if not optimization_df.empty:
            best = optimization_df.iloc[0]
            best_strategy = str(best["strategy_name"])
            base_top_n = int(best["top_n"])
            base_sector_cap = None if pd.isna(best["max_names_per_sector"]) else int(best["max_names_per_sector"])
            base_position_sizing = str(best["position_sizing"])
            base_cost_bps = float(best["total_cost_bps"])
            base_avoid_negative_news = bool(best["strong_negative_news_filter"])
            base_avoid_recent_downgrades = bool(best["recent_downgrade_filter"])

    price_path = config.processed_dir / "prices_all.csv"
    qqq_exists = False
    if price_path.exists():
        prices = load_dataframe(price_path)
        qqq_exists = "QQQ" in set(prices["ticker"])

    regime_specs = [
        {"regime_name": "none", "use_regime_filter": False, "regime_filter_type": "spy_200d", "regime_exposure": 1.0},
    ]
    for regime_filter_type in ["spy_50d", "spy_200d", "spy_50d_return_positive", "spy_21d_return_positive"]:
        for regime_exposure in [0.0, 0.5]:
            regime_specs.append(
                {
                    "regime_name": regime_filter_type,
                    "use_regime_filter": True,
                    "regime_filter_type": regime_filter_type,
                    "regime_exposure": regime_exposure,
                }
            )
    if qqq_exists:
        # QQQ-specific regime variants are skipped unless the feature panel includes QQQ-derived columns.
        pass

    rows: list[dict] = []
    for spec in regime_specs:
        weekly, _, _ = run_weekly_backtest(
            features=features,
            holding_period_days=5,
            benchmark=config.benchmark,
            top_n=base_top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=base_cost_bps,
            use_regime_filter=spec["use_regime_filter"],
            regime_exposure=0.0 if spec["regime_name"] == "none" else spec["regime_exposure"],
            use_analyst_filters=False,
            analyst_count_threshold=config.analyst_count_threshold,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            strategy_name=best_strategy,
            max_names_per_sector=base_sector_cap,
            use_inverse_vol_weighting=base_position_sizing == "inverse_volatility",
            position_sizing=base_position_sizing,
            regime_filter_type=spec["regime_filter_type"],
            avoid_strong_negative_news=base_avoid_negative_news,
            avoid_recent_downgrades=base_avoid_recent_downgrades,
            min_historical_rating_count=5,
        )
        full = _safe_metrics(weekly, 5)
        beat_windows = 0
        for label, start, end in WALK_FORWARD_WINDOWS:
            sliced = _slice_period(weekly, start, end)
            metrics = _safe_metrics(sliced, 5)
            beat_windows += int(bool(pd.notna(metrics["excess_total_return"]) and metrics["excess_total_return"] > 0))
        rows.append(
            {
                "strategy_name": best_strategy,
                "display_name": strategy_display_name(best_strategy),
                "top_n": base_top_n,
                "position_sizing": base_position_sizing,
                "total_cost_bps": base_cost_bps,
                "regime_name": spec["regime_name"],
                "regime_exposure_when_blocked": spec["regime_exposure"],
                "test_period_excess_vs_spy": _safe_metrics(_slice_period(weekly, "2025-01-01", "2025-12-31"), 5)["excess_total_return"],
                "walk_forward_windows_beating_spy": beat_windows,
                "max_drawdown": full["max_drawdown"],
                "turnover": full["average_turnover"],
                "percent_periods_invested": float((weekly["exposure"] > 0).mean()) if not weekly.empty else 0.0,
                "average_exposure": float(weekly["exposure"].mean()) if not weekly.empty else 0.0,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["walk_forward_windows_beating_spy", "test_period_excess_vs_spy", "max_drawdown", "percent_periods_invested"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    results_df["recommended_setting"] = False
    if not results_df.empty:
        baseline = results_df.loc[results_df["regime_name"] == "none"].iloc[0]
        recommended_idx = 0
        for idx, row in results_df.iterrows():
            improves = (
                row["walk_forward_windows_beating_spy"] > baseline["walk_forward_windows_beating_spy"]
                or (
                    row["walk_forward_windows_beating_spy"] == baseline["walk_forward_windows_beating_spy"]
                    and row["test_period_excess_vs_spy"] > baseline["test_period_excess_vs_spy"]
                    and row["max_drawdown"] >= baseline["max_drawdown"]
                )
            )
            if improves:
                recommended_idx = idx
                break
        results_df.loc[recommended_idx, "recommended_setting"] = True
        results_df["robustness_note"] = results_df.apply(
            lambda row: "improves_robustness"
            if row["walk_forward_windows_beating_spy"] > baseline["walk_forward_windows_beating_spy"]
            else ("reduces_exposure_only" if row["percent_periods_invested"] < baseline["percent_periods_invested"] else "no_clear_improvement"),
            axis=1,
        )

    save_dataframe(config.tables_dir / "regime_filter_test.csv", results_df)
    report_lines = [
        "# Regime Filter Test",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        f"- Base model tested: {strategy_display_name(best_strategy)}",
        f"- QQQ regime variants available: {qqq_exists}",
        "",
        "## Results",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    (config.reports_dir / "regime_filter_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved regime filter table to {config.tables_dir / 'regime_filter_test.csv'}")
    print(f"Saved regime filter report to {config.reports_dir / 'regime_filter_test.md'}")


if __name__ == "__main__":
    main()
