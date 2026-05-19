from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest
from src.config import Config
from src.no_snapshot_research import (
    WALK_FORWARD_WINDOWS,
    dataframe_to_markdown,
    fmt_pct,
    get_best_5d_config,
    load_features,
    run_benchmark_buy_hold,
    summarize_backtest,
)
from src.scoring import (
    apply_filters,
    get_strategy_filter_params,
    get_future_return_columns,
    score_rebalance_date,
    strategy_display_name,
)
from src.utils import save_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical."
SNAPSHOT_CAVEAT = "Snapshot analyst target models are excluded from the main historically safer ranking."
HISTORICAL_NOTE = "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date."
SENTIMENT_CAVEAT = "News sentiment depends on Alpha Vantage coverage and classification."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
REGIME_CAVEAT = "Regime filters were tested and are not recommended for the main model based on current results."
RESEARCH_CAVEAT = "This is research/paper trading only, not financial advice."


def _run_strategy(
    features: pd.DataFrame,
    config: Config,
    strategy_name: str,
    top_n: int,
    total_cost_bps: float,
    min_score_threshold: float | None,
    allow_cash: bool,
    max_names_per_sector: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    weekly, holdings, _ = run_weekly_backtest(
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
        strategy_name=strategy_name,
        max_names_per_sector=max_names_per_sector,
        position_sizing="equal_weight",
        min_historical_rating_count=5,
        min_score_threshold=min_score_threshold,
        allow_cash=allow_cash,
        min_holdings=1 if allow_cash else None,
    )
    summary = summarize_backtest(weekly, 5, strategy_name)
    summary["strategy_name"] = strategy_name
    summary["display_name"] = strategy_display_name(strategy_name)
    summary["top_n"] = top_n
    summary["total_cost_bps"] = total_cost_bps
    summary["min_score_threshold"] = min_score_threshold
    summary["allow_cash"] = allow_cash
    summary["average_holdings"] = summary.get("average_holdings", float("nan"))
    summary["average_percent_invested"] = float(weekly["percent_invested"].mean()) if not weekly.empty else 0.0
    summary["average_exposure"] = float(weekly["exposure"].mean()) if not weekly.empty else 0.0
    summary["average_cash_weight"] = float(weekly["cash_weight"].mean()) if not weekly.empty and "cash_weight" in weekly.columns else 0.0
    summary["selected_count_below_top_n_pct"] = float((weekly["selected_count"] < top_n).mean()) if not weekly.empty else 0.0
    return weekly, summary


def _selected_vs_non_selected(
    features: pd.DataFrame,
    config: Config,
    strategy_name: str,
    threshold: float | None,
    top_n: int,
) -> dict[str, float]:
    future_return_column, _, _ = get_future_return_columns(5)
    params = get_strategy_filter_params(
        strategy_name=strategy_name,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        min_historical_rating_count=5,
    )
    rows: list[dict[str, float]] = []
    unique_dates = sorted(pd.to_datetime(features["date"]).drop_duplicates())
    benchmark_dates = []
    for date in unique_dates:
        if not features.loc[(features["date"] == date) & (features["ticker"] == config.benchmark), future_return_column].isna().all():
            benchmark_dates.append(date)
    rebalance_dates = benchmark_dates[::5]
    for date in rebalance_dates:
        day = features.loc[(features["date"] == date) & (features["ticker"] != config.benchmark)].copy()
        qualified, _ = apply_filters(day, params=params, holding_period_days=5, benchmark=config.benchmark)
        scored = score_rebalance_date(qualified, strategy_name=strategy_name, use_analyst_filters=False).sort_values("score", ascending=False)
        passed = scored.copy()
        if threshold is not None:
            passed = passed.loc[pd.to_numeric(passed["score"], errors="coerce").fillna(-np.inf) > threshold].copy()
        selected = passed.head(top_n).copy()
        non_selected = scored.loc[~scored["ticker"].isin(selected["ticker"])].copy()
        spy_slice = features.loc[(features["date"] == date) & (features["ticker"] == config.benchmark), "future_5d_spy_return"]
        rows.append(
            {
                "selected_return": float(pd.to_numeric(selected[future_return_column], errors="coerce").mean()) if not selected.empty else 0.0,
                "non_selected_return": float(pd.to_numeric(non_selected[future_return_column], errors="coerce").mean()) if not non_selected.empty else 0.0,
                "spy_return": float(spy_slice.iloc[0]) if not spy_slice.empty else 0.0,
                "selected_count": len(selected),
            }
        )
    spread_df = pd.DataFrame(rows)
    return {
        "selected_minus_non_selected": float((spread_df["selected_return"] - spread_df["non_selected_return"]).mean()) if not spread_df.empty else float("nan"),
        "selected_minus_spy": float((spread_df["selected_return"] - spread_df["spy_return"]).mean()) if not spread_df.empty else float("nan"),
        "average_selected_count": float(spread_df["selected_count"].mean()) if not spread_df.empty else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)
    sector_cap = best_config["max_names_per_sector"]

    rows: list[dict[str, object]] = []

    spy_weekly = run_benchmark_buy_hold(features, 5, config.benchmark, config.benchmark)
    spy_summary = summarize_backtest(spy_weekly, 5, "spy")
    spy_summary["strategy_name"] = "spy"
    spy_summary["display_name"] = "SPY"
    spy_summary["top_n"] = 1
    spy_summary["total_cost_bps"] = 0.0
    spy_summary["min_score_threshold"] = None
    spy_summary["allow_cash"] = False
    spy_summary["average_percent_invested"] = 1.0
    spy_summary["average_exposure"] = 1.0
    spy_summary["average_cash_weight"] = 0.0
    rows.append(spy_summary)

    fixed_specs = [
        ("final_quant_5d_no_snapshot_no_sma_filter", 10, False, None, sector_cap),
        ("historical_rating_score_only_5d", 10, False, None, None),
        ("historical_rating_counts_plus_events", 10, False, None, None),
        ("final_quant_5d_no_recent_downgrade_filter_no_snapshot", 10, False, None, sector_cap),
    ]
    for strategy_name, top_n, allow_cash, threshold, max_names_per_sector in fixed_specs:
        for cost_bps in [10, 20]:
            _, summary = _run_strategy(
                features=features,
                config=config,
                strategy_name=strategy_name,
                top_n=top_n,
                total_cost_bps=cost_bps,
                min_score_threshold=threshold,
                allow_cash=allow_cash,
                max_names_per_sector=max_names_per_sector,
            )
            summary["strategy_family"] = "fixed"
            rows.append(summary)

    selective_specs = [
        ("final_quant_5d_selective_no_snapshot", sector_cap),
        ("historical_rating_score_selective_5d", None),
    ]
    for strategy_name, max_names_per_sector in selective_specs:
        for threshold in [None, 0.25, 0.50, 0.75, 1.00]:
            for top_n in [5, 10, 15]:
                for cost_bps in [10, 20]:
                    _, summary = _run_strategy(
                        features=features,
                        config=config,
                        strategy_name=strategy_name,
                        top_n=top_n,
                        total_cost_bps=cost_bps,
                        min_score_threshold=threshold,
                        allow_cash=True,
                        max_names_per_sector=max_names_per_sector,
                    )
                    summary["strategy_family"] = "selective"
                    rows.append(summary)

    results_df = pd.DataFrame(rows)
    results_df["rank_score"] = list(
        zip(
            -results_df["2024_h1_excess_return_vs_spy"].fillna(0.0) - results_df["2024_h2_excess_return_vs_spy"].fillna(0.0) - results_df["2025_excess_return_vs_spy"].fillna(0.0),
            -results_df["windows_beating_spy"].fillna(0),
            -results_df["2025_excess_return_vs_spy"].fillna(0.0),
            -results_df["max_drawdown"].fillna(-1.0),
            -results_df["average_exposure"].fillna(0.0),
        )
    )
    results_df = results_df.sort_values(
        [
            "2024_h1_excess_return_vs_spy",
            "2024_h2_excess_return_vs_spy",
            "2025_excess_return_vs_spy",
            "windows_beating_spy",
            "max_drawdown",
            "average_exposure",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    results_df["walk_forward_average_excess_vs_spy"] = results_df[
        ["2024_h1_excess_return_vs_spy", "2024_h2_excess_return_vs_spy", "2025_excess_return_vs_spy"]
    ].mean(axis=1)
    results_df = results_df.sort_values(
        [
            "walk_forward_average_excess_vs_spy",
            "windows_beating_spy",
            "2025_excess_return_vs_spy",
            "max_drawdown",
            "average_exposure",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "selective_strategy_test.csv", results_df)

    selective_rows = results_df.loc[results_df["strategy_name"] == "final_quant_5d_selective_no_snapshot"].copy()
    best_selective = selective_rows.iloc[0]
    selected_vs_non_selected = _selected_vs_non_selected(
        features=features,
        config=config,
        strategy_name="final_quant_5d_selective_no_snapshot",
        threshold=None if pd.isna(best_selective["min_score_threshold"]) else float(best_selective["min_score_threshold"]),
        top_n=int(best_selective["top_n"]),
    )
    allow_cash_false_weekly, allow_cash_false_summary = _run_strategy(
        features=features,
        config=config,
        strategy_name="final_quant_5d_selective_no_snapshot",
        top_n=int(best_selective["top_n"]),
        total_cost_bps=float(best_selective["total_cost_bps"]),
        min_score_threshold=None if pd.isna(best_selective["min_score_threshold"]) else float(best_selective["min_score_threshold"]),
        allow_cash=False,
        max_names_per_sector=sector_cap,
    )
    allow_cash_improves = bool(
        best_selective["sharpe_ratio"] >= allow_cash_false_summary["sharpe_ratio"]
        and best_selective["max_drawdown"] >= allow_cash_false_summary["max_drawdown"]
    )
    score_050_best = bool(
        (best_selective["min_score_threshold"] == 0.50)
        if pd.notna(best_selective["min_score_threshold"])
        else False
    )

    report_view = results_df[
        [
            "strategy_name",
            "display_name",
            "top_n",
            "total_cost_bps",
            "min_score_threshold",
            "allow_cash",
            "walk_forward_average_excess_vs_spy",
            "windows_beating_spy",
            "2025_excess_return_vs_spy",
            "max_drawdown",
            "average_holdings",
            "average_percent_invested",
        ]
    ].copy()
    for column in [
        "walk_forward_average_excess_vs_spy",
        "2025_excess_return_vs_spy",
        "max_drawdown",
        "average_holdings",
        "average_percent_invested",
    ]:
        report_view[column] = report_view[column].round(4)

    report_lines = [
        "# Selective Strategy Test",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        "## Findings",
        f"- Best selective configuration: `{best_selective['strategy_name']}` threshold={best_selective['min_score_threshold']}, top_n={int(best_selective['top_n'])}, cost_bps={int(best_selective['total_cost_bps'])}.",
        f"- Best selective walk-forward average excess vs SPY: {fmt_pct(best_selective['walk_forward_average_excess_vs_spy'])}.",
        f"- Best selective 2025 excess vs SPY: {fmt_pct(best_selective['2025_excess_return_vs_spy'])}.",
        f"- Best selective windows beating SPY: {int(best_selective['windows_beating_spy'])}/3.",
        f"- `score > 0.50` remains best: {score_050_best}.",
        f"- Allowing cash improves Sharpe/drawdown versus forced fill for the same setup: {allow_cash_improves}.",
        f"- Average holdings for best selective setup: {best_selective['average_holdings']:.2f}.",
        f"- Average percent invested for best selective setup: {fmt_pct(best_selective['average_percent_invested'])}.",
        f"- Selected stocks outperform non-selected stocks on average: {bool(selected_vs_non_selected['selected_minus_non_selected'] > 0)} ({fmt_pct(selected_vs_non_selected['selected_minus_non_selected'])}).",
        "",
        "## Results",
        "",
        dataframe_to_markdown(report_view.head(25)),
    ]
    (config.reports_dir / "selective_strategy_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved selective strategy table to {config.tables_dir / 'selective_strategy_test.csv'}")
    print(f"Saved selective strategy report to {config.reports_dir / 'selective_strategy_test.md'}")


if __name__ == "__main__":
    main()
