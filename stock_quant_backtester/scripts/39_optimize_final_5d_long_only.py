from __future__ import annotations

import argparse
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
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
STRATEGIES = [
    "final_quant_5d_no_snapshot_no_sma_filter",
    "historical_rating_counts_plus_events",
    "final_quant_5d_no_snapshot",
]
EXIT_OVERLAYS = [
    {"exit_overlay": "none"},
    {"exit_overlay": "stop_loss_8", "stop_loss_pct": 0.08},
    {"exit_overlay": "stop_loss_10", "stop_loss_pct": 0.10},
    {"exit_overlay": "trailing_stop_atr_2", "trailing_stop_atr_multiple": 2.0},
    {"exit_overlay": "trailing_stop_atr_3", "trailing_stop_atr_multiple": 3.0},
    {"exit_overlay": "stop_loss_8_take_profit_12", "stop_loss_pct": 0.08, "take_profit_pct": 0.12},
]
COST_GRID = [10, 20, 30]
WORKER_FEATURES: pd.DataFrame | None = None
WORKER_PRICES: pd.DataFrame | None = None
WORKER_EXIT_DATE_MAP: dict[pd.Timestamp, pd.Timestamp] | None = None
WORKER_BENCHMARK: str | None = None
WORKER_INITIAL_CAPITAL: float | None = None
WORKER_ANALYST_COUNT_THRESHOLD: int | None = None
WORKER_MIN_AVG_DOLLAR_VOLUME: float | None = None


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


def _build_exit_date_map(features: pd.DataFrame, benchmark: str, holding_period_days: int) -> dict[pd.Timestamp, pd.Timestamp]:
    benchmark_df = (
        features.loc[features["ticker"] == benchmark, ["date"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    dates = list(pd.to_datetime(benchmark_df["date"]).tolist())
    index_map = {date: idx for idx, date in enumerate(dates)}
    rebalance_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=benchmark)
    exit_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for date in rebalance_dates:
        idx = index_map[pd.Timestamp(date)]
        exit_idx = min(idx + holding_period_days, len(dates) - 1)
        exit_map[pd.Timestamp(date)] = pd.Timestamp(dates[exit_idx])
    return exit_map


def _simulate_overlay_return(
    price_path: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    atr_14: float,
    overlay: dict,
) -> float:
    path = price_path.loc[(price_path["date"] >= entry_date) & (price_path["date"] <= exit_date)].copy()
    if path.empty:
        return 0.0
    entry_close = float(path.iloc[0]["close"])
    scheduled_exit = float(path.iloc[-1]["close"])
    if overlay["exit_overlay"] == "none" or len(path) == 1:
        return scheduled_exit / entry_close - 1

    stop_loss_pct = overlay.get("stop_loss_pct")
    take_profit_pct = overlay.get("take_profit_pct")
    trailing_atr = overlay.get("trailing_stop_atr_multiple")
    highest_high = entry_close
    for row in path.iloc[1:].itertuples(index=False):
        low = float(row.low)
        high = float(row.high)
        close = float(row.close)
        if trailing_atr is not None and pd.notna(atr_14):
            highest_high = max(highest_high, high)
            trailing_stop = highest_high - trailing_atr * float(atr_14)
            if low <= trailing_stop:
                return trailing_stop / entry_close - 1
        if stop_loss_pct is not None:
            stop_price = entry_close * (1 - float(stop_loss_pct))
            if low <= stop_price:
                return stop_price / entry_close - 1
        if take_profit_pct is not None:
            take_price = entry_close * (1 + float(take_profit_pct))
            if high >= take_price:
                return take_price / entry_close - 1
        scheduled_exit = close
    return scheduled_exit / entry_close - 1


def _overlay_weekly_returns(
    holdings: pd.DataFrame,
    weekly: pd.DataFrame,
    prices: pd.DataFrame,
    exit_date_map: dict[pd.Timestamp, pd.Timestamp],
    overlay: dict,
) -> pd.DataFrame:
    price_groups = {ticker: frame.sort_values("date").copy() for ticker, frame in prices.groupby("ticker")}
    period_rows = []
    for row in weekly.itertuples(index=False):
        period_holdings = holdings.loc[holdings["date"] == pd.Timestamp(row.date)].copy()
        period_gross = 0.0
        for holding in period_holdings.itertuples(index=False):
            ticker_prices = price_groups.get(holding.ticker)
            if ticker_prices is None:
                realized = float(holding.future_return_used)
            else:
                realized = _simulate_overlay_return(
                    ticker_prices,
                    entry_date=pd.Timestamp(holding.date),
                    exit_date=exit_date_map[pd.Timestamp(holding.date)],
                    atr_14=float(getattr(holding, "atr_14", float("nan"))),
                    overlay=overlay,
                )
            period_gross += float(holding.weight) * realized
        period_rows.append(
            {
                "date": pd.Timestamp(row.date),
                "strategy_name": row.strategy_name,
                "selected_count": row.selected_count,
                "gross_return": period_gross,
                "turnover": row.turnover,
                "transaction_cost": row.transaction_cost,
                "net_return": period_gross - row.transaction_cost,
                "spy_return": row.spy_return,
                "excess_return": period_gross - row.transaction_cost - row.spy_return,
            }
        )
    overlay_df = pd.DataFrame(period_rows).sort_values("date").reset_index(drop=True)
    portfolio_value = 10000.0
    spy_value = 10000.0
    portfolio_values = []
    spy_values = []
    for row in overlay_df.itertuples(index=False):
        portfolio_value *= 1 + float(row.net_return)
        spy_value *= 1 + float(row.spy_return)
        portfolio_values.append(portfolio_value)
        spy_values.append(spy_value)
    overlay_df["portfolio_value"] = portfolio_values
    overlay_df["spy_value"] = spy_values
    overlay_df["exposure"] = 1.0
    return overlay_df


def _apply_transaction_cost(weekly: pd.DataFrame, total_cost_bps: float) -> pd.DataFrame:
    adjusted = weekly.copy()
    adjusted["transaction_cost"] = adjusted["turnover"].fillna(0.0) * float(total_cost_bps) / 10000.0
    adjusted["net_return"] = adjusted["gross_return"] - adjusted["transaction_cost"]
    adjusted["excess_return"] = adjusted["net_return"] - adjusted["spy_return"]

    portfolio_value = 10000.0
    spy_value = 10000.0
    portfolio_values = []
    spy_values = []
    for row in adjusted.itertuples(index=False):
        portfolio_value *= 1 + float(row.net_return)
        spy_value *= 1 + float(row.spy_return)
        portfolio_values.append(portfolio_value)
        spy_values.append(spy_value)
    adjusted["portfolio_value"] = portfolio_values
    adjusted["spy_value"] = spy_values
    return adjusted


def _robustness_label(row: pd.Series) -> str:
    if int(row["windows_beating_spy"]) < 2:
        return "weak"
    if float(row["average_walk_forward_excess_vs_spy"]) <= 0:
        return "weak"
    if float(row["worst_window_excess_vs_spy"]) < -0.10:
        return "fragile"
    if float(row["average_max_drawdown"]) > -0.20 and float(row["average_walk_forward_sharpe"]) > 1.0:
        return "robust"
    return "mixed"


def _init_worker(
    features_path: str,
    prices_path: str,
    benchmark: str,
    initial_capital: float,
    analyst_count_threshold: int,
    min_avg_dollar_volume: float,
) -> None:
    global WORKER_FEATURES
    global WORKER_PRICES
    global WORKER_EXIT_DATE_MAP
    global WORKER_BENCHMARK
    global WORKER_INITIAL_CAPITAL
    global WORKER_ANALYST_COUNT_THRESHOLD
    global WORKER_MIN_AVG_DOLLAR_VOLUME

    WORKER_FEATURES = load_dataframe(Path(features_path), parse_dates=["date"])
    WORKER_PRICES = load_dataframe(Path(prices_path), parse_dates=["date"])
    WORKER_EXIT_DATE_MAP = _build_exit_date_map(WORKER_FEATURES, benchmark, 5)
    WORKER_BENCHMARK = benchmark
    WORKER_INITIAL_CAPITAL = initial_capital
    WORKER_ANALYST_COUNT_THRESHOLD = analyst_count_threshold
    WORKER_MIN_AVG_DOLLAR_VOLUME = min_avg_dollar_volume


def _evaluate_core_config(task: tuple[str, int, str, int | None, bool, bool]) -> list[dict]:
    if WORKER_FEATURES is None or WORKER_PRICES is None or WORKER_EXIT_DATE_MAP is None:
        raise RuntimeError("Worker data is not initialized.")

    strategy_name, top_n, position_sizing, max_names_per_sector, strong_negative_news_filter, recent_downgrade_filter = task
    weekly, holdings, _ = run_weekly_backtest(
        features=WORKER_FEATURES,
        holding_period_days=5,
        benchmark=WORKER_BENCHMARK,
        top_n=top_n,
        initial_capital=WORKER_INITIAL_CAPITAL,
        transaction_cost_bps=0,
        use_regime_filter=False,
        regime_exposure=0.0,
        use_analyst_filters=False,
        analyst_count_threshold=WORKER_ANALYST_COUNT_THRESHOLD,
        min_avg_dollar_volume=WORKER_MIN_AVG_DOLLAR_VOLUME,
        strategy_name=strategy_name,
        max_names_per_sector=max_names_per_sector,
        use_inverse_vol_weighting=position_sizing == "inverse_volatility",
        position_sizing=position_sizing,
        avoid_strong_negative_news=strong_negative_news_filter,
        avoid_recent_downgrades=recent_downgrade_filter,
        min_historical_rating_count=5,
    )
    holdings = holdings.merge(
        WORKER_FEATURES[["date", "ticker", "atr_14"]].drop_duplicates(["date", "ticker"]),
        on=["date", "ticker"],
        how="left",
    )

    rows: list[dict] = []
    for overlay in EXIT_OVERLAYS:
        effective_weekly_base = (
            weekly if overlay["exit_overlay"] == "none" else _overlay_weekly_returns(holdings, weekly, WORKER_PRICES, WORKER_EXIT_DATE_MAP, overlay)
        )
        for total_cost_bps in COST_GRID:
            effective_weekly = _apply_transaction_cost(effective_weekly_base, total_cost_bps)
            window_rows = []
            for window_label, start, end in WALK_FORWARD_WINDOWS:
                metrics = _safe_metrics(_slice_period(effective_weekly, start, end), 5)
                window_rows.append(
                    {
                        "window_label": window_label,
                        "excess": metrics["excess_total_return"],
                        "drawdown": metrics["max_drawdown"],
                        "sharpe": metrics["sharpe_ratio"],
                        "beat_spy": bool(pd.notna(metrics["excess_total_return"]) and metrics["excess_total_return"] > 0),
                    }
                )
            walk_df = pd.DataFrame(window_rows)
            test_metrics = _safe_metrics(_slice_period(effective_weekly, "2025-01-01", "2025-12-31"), 5)
            full_metrics = _safe_metrics(effective_weekly, 5)
            row = {
                "strategy_name": strategy_name,
                "display_name": strategy_display_name(strategy_name),
                "holding_period_days": 5,
                "top_n": top_n,
                "position_sizing": position_sizing,
                "total_cost_bps": total_cost_bps,
                "max_names_per_sector": max_names_per_sector,
                "exit_overlay": overlay["exit_overlay"],
                "strong_negative_news_filter": strong_negative_news_filter,
                "recent_downgrade_filter": recent_downgrade_filter,
                "regime_filter": "none",
                "long_short": False,
                "average_walk_forward_excess_vs_spy": float(walk_df["excess"].mean()),
                "windows_beating_spy": int(walk_df["beat_spy"].sum()),
                "worst_window_excess_vs_spy": float(walk_df["excess"].min()),
                "average_max_drawdown": float(walk_df["drawdown"].mean()),
                "average_walk_forward_sharpe": float(walk_df["sharpe"].mean()),
                "test_2025_excess_vs_spy": test_metrics["excess_total_return"],
                "test_2025_return": test_metrics["total_return"],
                "full_period_max_drawdown": full_metrics["max_drawdown"],
                "average_turnover": full_metrics["average_turnover"],
                "average_holdings": full_metrics["average_selected_count"],
                "is_baseline": bool(
                    strategy_name == "final_quant_5d_no_snapshot_no_sma_filter"
                    and top_n == 10
                    and position_sizing == "equal_weight"
                    and total_cost_bps == 10
                    and max_names_per_sector is None
                    and overlay["exit_overlay"] == "none"
                    and strong_negative_news_filter is False
                    and recent_downgrade_filter is False
                ),
            }
            row["meets_selection_rule"] = bool(row["windows_beating_spy"] >= 2)
            row["robustness_flag"] = _robustness_label(pd.Series(row))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    core_tasks = list(
        product(
            STRATEGIES,
            [5, 10, 15, 20],
            ["equal_weight", "inverse_volatility", "score_weighted", "score_over_volatility"],
            [None, 3, 4],
            [False, True],
            [False, True],
        )
    )

    rows: list[dict] = []
    worker_count = min(4, os.cpu_count() or 1)
    with mp.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(
            str(features_path),
            str(config.processed_dir / "prices_all.csv"),
            config.benchmark,
            config.initial_capital,
            config.analyst_count_threshold,
            config.min_avg_dollar_volume,
        ),
    ) as pool:
        for idx, result_rows in enumerate(pool.imap_unordered(_evaluate_core_config, core_tasks), start=1):
            rows.extend(result_rows)
            if idx % 24 == 0 or idx == len(core_tasks):
                print(f"Completed {idx}/{len(core_tasks)} base configs")

    results_df = pd.DataFrame(rows).sort_values(
        [
            "meets_selection_rule",
            "average_walk_forward_excess_vs_spy",
            "windows_beating_spy",
            "test_2025_excess_vs_spy",
            "average_max_drawdown",
            "average_turnover",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    results_df["selected_best"] = False
    if not results_df.empty:
        results_df.loc[0, "selected_best"] = True

    save_dataframe(config.tables_dir / "final_5d_long_only_optimization.csv", results_df)
    baseline_row = results_df.loc[results_df["is_baseline"]].iloc[0] if not results_df.loc[results_df["is_baseline"]].empty else None
    best = results_df.iloc[0] if not results_df.empty else None
    report_lines = [
        "# Final 5D Long Only Optimization",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
    ]
    if baseline_row is not None:
        report_lines.append(
            f"- Baseline row: {baseline_row['display_name']}, top_n=10, equal_weight, total_cost_bps=10, max_names_per_sector=None, exit_overlay=none, no regime filter, long_short=false"
        )
    if best is not None:
        report_lines.extend(
            [
                f"- Best configuration: {best['display_name']} / top_n={int(best['top_n'])} / position_sizing={best['position_sizing']} / total_cost_bps={int(best['total_cost_bps'])} / sector_cap={best['max_names_per_sector']} / exit_overlay={best['exit_overlay']}",
                f"- Windows beating SPY: {int(best['windows_beating_spy'])}/3",
                f"- Average walk-forward excess vs SPY: {float(best['average_walk_forward_excess_vs_spy']):.2%}",
                f"- Worst window excess vs SPY: {float(best['worst_window_excess_vs_spy']):.2%}",
                f"- 2025 excess vs SPY: {float(best['test_2025_excess_vs_spy']):.2%}",
                f"- Average walk-forward Sharpe: {float(best['average_walk_forward_sharpe']):.2f}",
                f"- Average walk-forward drawdown: {float(best['average_max_drawdown']):.2%}",
                f"- Average turnover: {float(best['average_turnover']):.4f}",
                f"- Robustness assessment: {best['robustness_flag']}",
            ]
        )
    report_lines.extend(["", "## Results", "", _dataframe_to_markdown(results_df.head(50).round(6))])
    (config.reports_dir / "final_5d_long_only_optimization.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved optimization table to {config.tables_dir / 'final_5d_long_only_optimization.csv'}")
    print(f"Saved optimization report to {config.reports_dir / 'final_5d_long_only_optimization.md'}")


if __name__ == "__main__":
    main()
