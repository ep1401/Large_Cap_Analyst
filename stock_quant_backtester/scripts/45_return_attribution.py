from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.no_snapshot_research import (
    dataframe_to_markdown,
    fmt_pct,
    get_best_5d_config,
    load_features,
    run_custom_weekly_backtest,
    build_final_quant_5d_definition,
)
from src.utils import save_dataframe


def _compound_return(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float((1 + pd.to_numeric(series, errors="coerce").fillna(0.0)).prod() - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)

    weekly, holdings, _ = run_custom_weekly_backtest(
        features=features,
        definition=build_final_quant_5d_definition(),
        holding_period_days=5,
        benchmark=config.benchmark,
        top_n=int(best_config["top_n"]),
        transaction_cost_bps=float(best_config["total_cost_bps"]),
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        max_names_per_sector=best_config["max_names_per_sector"],
        position_sizing=str(best_config["position_sizing"]),
        max_single_name_weight=float(best_config["max_single_name_weight"]),
    )

    weekly = weekly.copy()
    weekly["month"] = pd.to_datetime(weekly["date"]).dt.to_period("M").astype(str)
    weekly["quarter"] = pd.to_datetime(weekly["date"]).dt.to_period("Q").astype(str)
    holdings = holdings.copy()
    holdings["gross_contribution"] = holdings["weight"] * holdings["future_return_used"]
    holdings["excess_contribution"] = holdings["weight"] * holdings["future_excess_return_used"]
    holdings["month"] = pd.to_datetime(holdings["date"]).dt.to_period("M").astype(str)
    holdings["quarter"] = pd.to_datetime(holdings["date"]).dt.to_period("Q").astype(str)

    month_df = (
        weekly.groupby("month", as_index=False)
        .agg(
            strategy_return=("net_return", _compound_return),
            spy_return=("spy_return", _compound_return),
            excess_return=("excess_return", "sum"),
            periods=("date", "count"),
        )
        .sort_values("month")
    )
    quarter_df = (
        weekly.groupby("quarter", as_index=False)
        .agg(
            strategy_return=("net_return", _compound_return),
            spy_return=("spy_return", _compound_return),
            excess_return=("excess_return", "sum"),
            periods=("date", "count"),
        )
        .sort_values("quarter")
    )
    ticker_df = (
        holdings.groupby(["ticker", "sector"], dropna=False, as_index=False)
        .agg(
            gross_contribution=("gross_contribution", "sum"),
            excess_contribution=("excess_contribution", "sum"),
            periods_held=("date", "count"),
            avg_weight=("weight", "mean"),
            avg_score=("score", "mean"),
        )
        .sort_values("excess_contribution", ascending=False)
    )
    sector_df = pd.DataFrame()
    if "sector" in holdings.columns:
        sector_df = (
            holdings.groupby("sector", dropna=False, as_index=False)
            .agg(
                gross_contribution=("gross_contribution", "sum"),
                excess_contribution=("excess_contribution", "sum"),
                periods_held=("date", "count"),
            )
            .sort_values("excess_contribution", ascending=False)
        )

    top_winners = holdings.sort_values("excess_contribution", ascending=False).head(10)[
        ["date", "ticker", "sector", "weight", "score", "future_return_used", "future_excess_return_used", "excess_contribution"]
    ].copy()
    top_losers = holdings.sort_values("excess_contribution", ascending=True).head(10)[
        ["date", "ticker", "sector", "weight", "score", "future_return_used", "future_excess_return_used", "excess_contribution"]
    ].copy()

    save_dataframe(config.tables_dir / "return_attribution_by_month.csv", month_df)
    save_dataframe(config.tables_dir / "return_attribution_by_ticker.csv", ticker_df)

    positive_ticker_excess = ticker_df.loc[ticker_df["excess_contribution"] > 0, "excess_contribution"].sum()
    positive_month_excess = month_df.loc[month_df["excess_return"] > 0, "excess_return"].sum()
    top5_ticker_share = (
        float(ticker_df.loc[ticker_df["excess_contribution"] > 0].head(5)["excess_contribution"].sum() / positive_ticker_excess)
        if positive_ticker_excess
        else float("nan")
    )
    top3_month_share = (
        float(month_df.loc[month_df["excess_return"] > 0].sort_values("excess_return", ascending=False).head(3)["excess_return"].sum() / positive_month_excess)
        if positive_month_excess
        else float("nan")
    )
    broad_based = bool(
        pd.notna(top5_ticker_share)
        and pd.notna(top3_month_share)
        and top5_ticker_share < 0.60
        and top3_month_share < 0.60
    )

    report_lines = [
        "# Return Attribution Report",
        "",
        "- Strategy analyzed: `final_quant_5d_no_snapshot_no_sma_filter`.",
        f"- Percent of total excess return from top 5 tickers: {fmt_pct(top5_ticker_share)}.",
        f"- Percent of total excess return from top 3 months: {fmt_pct(top3_month_share)}.",
        f"- Performance looks broad-based rather than concentrated: {broad_based}.",
        f"- One month or ticker appears to explain most of the edge: {bool(pd.notna(top5_ticker_share) and (top5_ticker_share >= 0.60 or top3_month_share >= 0.60))}.",
        "",
        "## Excess Return By Month",
        "",
        dataframe_to_markdown(month_df.round(6)),
        "",
        "## Excess Return By Quarter",
        "",
        dataframe_to_markdown(quarter_df.round(6)),
        "",
        "## Excess Contribution By Sector",
        "",
        dataframe_to_markdown(sector_df.round(6) if not sector_df.empty else pd.DataFrame({"sector": ["n/a"]})),
        "",
        "## Top 10 Winning Holdings",
        "",
        dataframe_to_markdown(top_winners.round(6)),
        "",
        "## Top 10 Losing Holdings",
        "",
        dataframe_to_markdown(top_losers.round(6)),
    ]
    (config.reports_dir / "return_attribution_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved monthly attribution table to {config.tables_dir / 'return_attribution_by_month.csv'}")
    print(f"Saved ticker attribution table to {config.tables_dir / 'return_attribution_by_ticker.csv'}")
    print(f"Saved attribution report to {config.reports_dir / 'return_attribution_report.md'}")


if __name__ == "__main__":
    main()
