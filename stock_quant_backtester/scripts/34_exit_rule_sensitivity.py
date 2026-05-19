from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import get_future_return_columns, strategy_display_name
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
OVERLAYS = [
    {"overlay_name": "none"},
    {"overlay_name": "stop_loss_5pct", "stop_loss_pct": 0.05},
    {"overlay_name": "stop_loss_8pct", "stop_loss_pct": 0.08},
    {"overlay_name": "stop_loss_10pct", "stop_loss_pct": 0.10},
    {"overlay_name": "take_profit_8pct", "take_profit_pct": 0.08},
    {"overlay_name": "take_profit_12pct", "take_profit_pct": 0.12},
    {"overlay_name": "take_profit_15pct", "take_profit_pct": 0.15},
    {"overlay_name": "stop_8_take_12", "stop_loss_pct": 0.08, "take_profit_pct": 0.12},
    {"overlay_name": "stop_8_take_15", "stop_loss_pct": 0.08, "take_profit_pct": 0.15},
    {"overlay_name": "sell_half_take_12", "take_profit_pct": 0.12, "sell_half_at_take_profit": True},
    {"overlay_name": "trailing_stop_atr_2", "trailing_stop_atr_multiple": 2.0},
    {"overlay_name": "trailing_stop_atr_3", "trailing_stop_atr_multiple": 3.0},
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
    if overlay["overlay_name"] == "none" or len(path) == 1:
        return scheduled_exit / entry_close - 1

    stop_loss_pct = overlay.get("stop_loss_pct")
    take_profit_pct = overlay.get("take_profit_pct")
    sell_half = bool(overlay.get("sell_half_at_take_profit", False))
    trailing_atr = overlay.get("trailing_stop_atr_multiple")

    highest_high = entry_close
    half_taken = False
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
                if sell_half and not half_taken:
                    half_taken = True
                    return 0.5 * (take_price / entry_close - 1) + 0.5 * (scheduled_exit / entry_close - 1)
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
                    atr_14=float(getattr(holding, "atr_14", float("nan"))) if hasattr(holding, "atr_14") else float("nan"),
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


def _load_models(config: Config) -> list[dict]:
    return [
        {
            "strategy_name": "final_quant_5d_no_snapshot_no_sma_filter",
            "top_n": config.top_n,
            "max_names_per_sector": None,
            "position_sizing": "equal_weight",
            "transaction_cost_bps": config.transaction_cost_bps,
            "avoid_strong_negative_news": False,
            "avoid_recent_downgrades": False,
        },
        {
            "strategy_name": "historical_rating_counts_plus_events",
            "top_n": config.top_n,
            "max_names_per_sector": None,
            "position_sizing": "equal_weight",
            "transaction_cost_bps": config.transaction_cost_bps,
            "avoid_strong_negative_news": False,
            "avoid_recent_downgrades": False,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    prices = load_dataframe(config.processed_dir / "prices_all.csv", parse_dates=["date"])
    exit_date_map = _build_exit_date_map(features, config.benchmark, 5)
    models = _load_models(config)

    rows: list[dict] = []
    for model in models:
        weekly, holdings, _ = run_weekly_backtest(
            features=features,
            holding_period_days=5,
            benchmark=config.benchmark,
            top_n=model["top_n"],
            initial_capital=config.initial_capital,
            transaction_cost_bps=model["transaction_cost_bps"],
            use_regime_filter=False,
            regime_exposure=0.0,
            use_analyst_filters=False,
            analyst_count_threshold=config.analyst_count_threshold,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            strategy_name=model["strategy_name"],
            max_names_per_sector=model["max_names_per_sector"],
            use_inverse_vol_weighting=model["position_sizing"] == "inverse_volatility",
            position_sizing=model["position_sizing"],
            avoid_strong_negative_news=model["avoid_strong_negative_news"],
            avoid_recent_downgrades=model["avoid_recent_downgrades"],
            min_historical_rating_count=5,
        )
        holdings = holdings.merge(
            features[["date", "ticker", "atr_14"]].drop_duplicates(["date", "ticker"]),
            on=["date", "ticker"],
            how="left",
        )
        baseline_metrics = None
        baseline_drawdown = None
        baseline_sharpe = None
        baseline_test_excess = None
        for overlay in OVERLAYS:
            overlay_weekly = _overlay_weekly_returns(holdings, weekly, prices, exit_date_map, overlay)
            full = _safe_metrics(overlay_weekly, 5)
            test = _safe_metrics(_slice_period(overlay_weekly, "2025-01-01", "2025-12-31"), 5)
            walk_rows = []
            for window_label, start, end in WALK_FORWARD_WINDOWS:
                window_metrics = _safe_metrics(_slice_period(overlay_weekly, start, end), 5)
                walk_rows.append(
                    {
                        "window_label": window_label,
                        "excess": window_metrics["excess_total_return"],
                        "drawdown": window_metrics["max_drawdown"],
                        "sharpe": window_metrics["sharpe_ratio"],
                        "beat_spy": bool(pd.notna(window_metrics["excess_total_return"]) and window_metrics["excess_total_return"] > 0),
                    }
                )
            walk_df = pd.DataFrame(walk_rows)
            if overlay["overlay_name"] == "none":
                baseline_metrics = full
                baseline_drawdown = full["max_drawdown"]
                baseline_sharpe = full["sharpe_ratio"]
                baseline_test_excess = test["excess_total_return"]
            rows.append(
                {
                    "strategy_name": model["strategy_name"],
                    "display_name": strategy_display_name(model["strategy_name"]),
                    "top_n": model["top_n"],
                    "position_sizing": model["position_sizing"],
                    "overlay_name": overlay["overlay_name"],
                    "stop_loss_pct": overlay.get("stop_loss_pct"),
                    "take_profit_pct": overlay.get("take_profit_pct"),
                    "sell_half_at_take_profit": bool(overlay.get("sell_half_at_take_profit", False)),
                    "trailing_stop_atr_multiple": overlay.get("trailing_stop_atr_multiple"),
                    "full_period_return": full["total_return"],
                    "test_period_return": test["total_return"],
                    "test_period_excess_vs_spy": test["excess_total_return"],
                    "walk_forward_average_excess_vs_spy": float(walk_df["excess"].mean()),
                    "walk_forward_windows_beating_spy": int(walk_df["beat_spy"].sum()),
                    "walk_forward_average_drawdown": float(walk_df["drawdown"].mean()),
                    "sharpe": full["sharpe_ratio"],
                    "max_drawdown": full["max_drawdown"],
                    "average_turnover": full["average_turnover"],
                    "average_holdings": full["average_selected_count"],
                    "is_baseline": bool(model["strategy_name"] == "final_quant_5d_no_snapshot_no_sma_filter" and overlay["overlay_name"] == "none"),
                    "improves_return_vs_none": False if baseline_test_excess is None else bool(test["excess_total_return"] > baseline_test_excess),
                    "improves_drawdown_vs_none": False if baseline_drawdown is None else bool(full["max_drawdown"] > baseline_drawdown),
                    "improves_sharpe_vs_none": False if baseline_sharpe is None else bool(full["sharpe_ratio"] > baseline_sharpe),
                }
            )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        recommended_flags = []
        for strategy_name, group in results_df.groupby("strategy_name"):
            group = group.copy()
            base = group.loc[group["overlay_name"] == "none"].iloc[0]
            preferred = group.loc[
                (group["improves_drawdown_vs_none"] | group["improves_sharpe_vs_none"])
                & (group["test_period_excess_vs_spy"] >= base["test_period_excess_vs_spy"])
            ]
            preferred_name = preferred.sort_values(
                ["test_period_excess_vs_spy", "sharpe", "max_drawdown"],
                ascending=[False, False, False],
            ).iloc[0]["overlay_name"] if not preferred.empty else "none"
            for row in group.itertuples(index=False):
                recommended_flags.append({"strategy_name": strategy_name, "overlay_name": row.overlay_name, "recommended_overlay": row.overlay_name == preferred_name})
        flags_df = pd.DataFrame(recommended_flags)
        results_df = results_df.merge(flags_df, on=["strategy_name", "overlay_name"], how="left")

    save_dataframe(config.tables_dir / "exit_rule_sensitivity.csv", results_df.sort_values(["strategy_name", "overlay_name"]))

    take_profit_underperformed = bool(
        not results_df.loc[results_df["overlay_name"].str.contains("take_profit", na=False) & results_df["improves_return_vs_none"]].any().any()
    ) if not results_df.empty else False
    no_overlay_best = bool(results_df["recommended_overlay"].fillna(False).sum() == len(results_df["strategy_name"].unique())) if not results_df.empty else True
    best_walk = results_df.sort_values(
        ["walk_forward_average_excess_vs_spy", "walk_forward_windows_beating_spy", "test_period_excess_vs_spy", "max_drawdown"],
        ascending=[False, False, False, False],
    ).iloc[0] if not results_df.empty else None
    best_drawdown = results_df.sort_values("max_drawdown", ascending=False).iloc[0] if not results_df.empty else None
    report_lines = [
        "# Exit Rule Sensitivity",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        f"- Baseline row: final_quant_5d_no_snapshot_no_sma_filter, top_n=10, equal_weight, 10 bps, no exit overlay, no regime filter",
        f"- Best exit rule by walk-forward excess: {best_walk['strategy_name']} / {best_walk['overlay_name']}" if best_walk is not None else "- Best exit rule by walk-forward excess: n/a",
        f"- Best exit rule by max drawdown: {best_drawdown['strategy_name']} / {best_drawdown['overlay_name']}" if best_drawdown is not None else "- Best exit rule by max drawdown: n/a",
        f"- Whether exits improve return in some cases: {bool(results_df['improves_return_vs_none'].any()) if not results_df.empty else False}",
        f"- Whether exits improve drawdown in some cases: {bool(results_df['improves_drawdown_vs_none'].any()) if not results_df.empty else False}",
        f"- Whether exits improve Sharpe in some cases: {bool(results_df['improves_sharpe_vs_none'].any()) if not results_df.empty else False}",
        f"- Whether take-profit appears to cut winners: {take_profit_underperformed}",
        f"- Whether stop-loss tends to reduce drawdown but hurt return: {bool(results_df.loc[results_df['overlay_name'].str.contains('stop_loss', na=False) & results_df['improves_drawdown_vs_none'] & ~results_df['improves_return_vs_none']].shape[0] > 0) if not results_df.empty else False}",
        f"- Whether no exit overlay remains best: {no_overlay_best}",
        "",
        "## Results",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    (config.reports_dir / "exit_rule_sensitivity.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved exit rule table to {config.tables_dir / 'exit_rule_sensitivity.csv'}")
    print(f"Saved exit rule report to {config.reports_dir / 'exit_rule_sensitivity.md'}")


if __name__ == "__main__":
    main()
