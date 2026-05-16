from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.build_features import build_feature_panel
from src.config import Config
from src.fetch_alpha_vantage_news import build_monthly_windows, summarize_request_plan
from src.metrics import calculate_performance_metrics
from src.scoring import SNAPSHOT_ANALYST_STRATEGIES, strategy_analyst_data_mode
from src.universe import get_tickers
from src.utils import load_dataframe, save_dataframe


ANALYST_SNAPSHOT_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged across historical dates "
    "unless true point-in-time analyst history is provided. These results should be treated as research exploration, not a "
    "valid historical analyst-signal backtest."
)
SENTIMENT_CAVEAT = (
    "News sentiment results depend on Alpha Vantage coverage, ticker relevance scoring, publication timestamps, and provider "
    "sentiment classification."
)
BACKTEST_CAVEAT = "This is a historical research backtest, not financial advice and not a live trading system."
ONE_YEAR_CAVEAT = (
    "Because this run uses a one-year default window, results may be noisy and should be validated over longer periods once "
    "cached data is available."
)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_spy_weekly(features: pd.DataFrame, config: Config, holding_period_days: int) -> pd.DataFrame:
    future_spy_map = {5: "future_5d_spy_return", 21: "future_21d_spy_return", 63: "future_63d_spy_return"}
    rebalance_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=config.benchmark)
    spy_value = config.initial_capital
    rows: list[dict] = []
    for date in rebalance_dates:
        spy_return = float(
            features.loc[
                (features["ticker"] == config.benchmark) & (features["date"] == date),
                future_spy_map[holding_period_days],
            ].iloc[0]
        )
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": "SPY",
                "selected_count": 1,
                "qualified_count": 1,
                "gross_return": spy_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": spy_return,
                "spy_return": spy_return,
                "excess_return": 0.0,
                "portfolio_value": spy_value,
                "spy_value": spy_value,
                "exposure": 1.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _metrics_row(
    strategy_name: str,
    weekly: pd.DataFrame,
    *,
    holding_period_days: int,
    top_n: int,
    max_names_per_sector: int | None,
) -> dict:
    metrics = calculate_performance_metrics(weekly, holding_period_days=holding_period_days)
    return {
        "strategy_name": strategy_name,
        "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
        "holding_period_days": holding_period_days,
        "top_n": top_n,
        "max_names_per_sector": max_names_per_sector,
        "evaluation_mode": "short_sample_evaluation",
        "total_return": metrics["total_return"],
        "spy_total_return": metrics["spy_total_return"],
        "excess_return_vs_spy": metrics["excess_total_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "periods_beating_spy": metrics["weeks_beating_spy"],
        "average_turnover": metrics["average_turnover"],
        "average_holdings": metrics["average_selected_count"],
        "number_of_rebalance_periods": metrics["number_of_rebalance_periods"],
    }


def _sentiment_coverage(features: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    rows = []
    for date, day in features.loc[features["ticker"] != benchmark].groupby("date"):
        total = len(day)
        with_news = int((day["article_count_7d"] > 0).sum()) if "article_count_7d" in day.columns else 0
        rows.append(
            {
                "date": pd.to_datetime(date),
                "coverage_pct_7d": (with_news / total) if total else 0.0,
                "average_article_count_7d": float(day["article_count_7d"].mean()) if "article_count_7d" in day.columns else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force-rebuild-features", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    start_date = args.start_date or config.start_date
    end_date = args.end_date or config.end_date
    date_label = f"{start_date}_{end_date}"
    print(config.describe_analysis_windows())

    required_processed = [
        config.processed_dir / "prices_all.csv",
        config.processed_dir / "analyst_features.csv",
    ]
    for path in required_processed:
        if not path.exists():
            raise SystemExit(f"Missing required processed file: {path}. Run the corresponding fetch script first.")

    sentiment_daily_path = config.processed_dir / "news_sentiment_daily.csv"
    av_windows = build_monthly_windows(
        tickers=get_tickers(config.universe_path),
        start_date=config.sentiment_start_date,
        end_date=config.sentiment_end_date,
        raw_news_dir=config.raw_dir / "news" / "alpha_vantage",
    )
    av_plan = summarize_request_plan(
        av_windows,
        force=False,
        cache_enabled=config.cache_enabled,
        requests_per_minute=config.alpha_vantage_requests_per_minute,
    )
    print(f"Alpha Vantage cached files: {av_plan['cached_requests']} / {av_plan['total_possible_requests']}")
    if not sentiment_daily_path.exists() and av_plan["missing_requests"] > 0:
        raise SystemExit(
            f"Missing sentiment files and {av_plan['missing_requests']} ticker-month Alpha Vantage files are still uncached. "
            "Run `python scripts/12_fetch_alpha_vantage_news.py` first."
        )

    features_path = config.final_dir / "features_panel.csv"
    if args.force_rebuild_features or not features_path.exists():
        build_feature_panel(
            prices_path=config.processed_dir / "prices_all.csv",
            universe_path=config.universe_path,
            analyst_path=config.processed_dir / "analyst_features.csv",
            sentiment_path=sentiment_daily_path,
            historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
            historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
            historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
            historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
            output_path=features_path,
            benchmark_ticker=config.benchmark,
            use_current_snapshot_analyst=True,
        )

    features = load_dataframe(features_path, parse_dates=["date"])
    features = features.loc[(features["date"] >= pd.Timestamp(start_date)) & (features["date"] < pd.Timestamp(end_date))].copy()
    if features.empty:
        raise SystemExit("No features available inside the requested one-year analysis window.")

    has_sentiment = (
        "sentiment_data_mode" in features.columns
        and not features["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all()
    )
    has_historical_ratings = (
        "historical_rating_count_data_available" in features.columns
        and features["historical_rating_count_data_available"].fillna(False).any()
    )

    strategy_specs = [
        {"strategy_name": "analyst_snapshot_model", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "technical_only", "use_analyst_filters": False, "max_names_per_sector": None},
        {"strategy_name": "technical_momentum_model", "use_analyst_filters": False, "max_names_per_sector": None},
        {"strategy_name": "full_model", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "strict_checklist_model", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "final_quant_model_1y", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "final_quant_model_1y_no_sentiment", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "final_quant_model_1y_sentiment_risk_filter", "use_analyst_filters": True, "max_names_per_sector": None},
        {"strategy_name": "final_quant_model_1y_sector_capped", "use_analyst_filters": True, "max_names_per_sector": 3},
    ]
    skipped_models: list[str] = []
    if has_sentiment and has_historical_ratings:
        strategy_specs.append({"strategy_name": "final_quant_model_1y_no_snapshot", "use_analyst_filters": False, "max_names_per_sector": None})
    else:
        skipped_models.append("final_quant_model_1y_no_snapshot (requires sentiment and historical analyst rating-count features)")

    comparison_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []
    best_weekly_by_strategy: dict[str, pd.DataFrame] = {}

    for holding_period_days in [5, 21, 63]:
        spy_weekly = _build_spy_weekly(features, config, holding_period_days)
        comparison_rows.append(
            _metrics_row("SPY", spy_weekly, holding_period_days=holding_period_days, top_n=1, max_names_per_sector=None)
        )
        curve_rows.append(spy_weekly.assign(strategy_name="SPY", holding_period_days=holding_period_days, top_n=1))

        for top_n in [5, 10, 20]:
            for spec in strategy_specs:
                strategy_name = spec["strategy_name"]
                try:
                    weekly, _, _ = run_weekly_backtest(
                        features=features,
                        holding_period_days=holding_period_days,
                        benchmark=config.benchmark,
                        top_n=top_n,
                        initial_capital=config.initial_capital,
                        transaction_cost_bps=config.transaction_cost_bps,
                        use_regime_filter=False,
                        regime_exposure=0.0,
                        use_analyst_filters=spec["use_analyst_filters"],
                        analyst_count_threshold=config.analyst_count_threshold,
                        min_avg_dollar_volume=config.min_avg_dollar_volume,
                        strategy_name=strategy_name,
                        max_names_per_sector=spec["max_names_per_sector"],
                        min_grade_events_90d=1,
                    )
                except ValueError as exc:
                    skipped_models.append(f"{strategy_name} ({exc})")
                    continue

                row = _metrics_row(
                    strategy_name,
                    weekly,
                    holding_period_days=holding_period_days,
                    top_n=top_n,
                    max_names_per_sector=spec["max_names_per_sector"],
                )
                row["sentiment_coverage"] = float(
                    features.loc[features["ticker"] != config.benchmark, "article_count_7d"].gt(0).mean()
                ) if "article_count_7d" in features.columns else 0.0
                comparison_rows.append(row)

                key = strategy_name
                existing = best_weekly_by_strategy.get(key)
                if existing is None:
                    best_weekly_by_strategy[key] = weekly.assign(strategy_name=strategy_name, holding_period_days=holding_period_days, top_n=top_n)
                else:
                    existing_metrics = calculate_performance_metrics(existing, int(existing["holding_period_days"].iloc[0]))
                    if row["excess_return_vs_spy"] > existing_metrics["excess_total_return"]:
                        best_weekly_by_strategy[key] = weekly.assign(
                            strategy_name=strategy_name,
                            holding_period_days=holding_period_days,
                            top_n=top_n,
                        )

    comparison_df = pd.DataFrame(comparison_rows).drop_duplicates().sort_values(
        ["excess_return_vs_spy", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    generic_csv = config.tables_dir / "final_quant_model_1y_comparison.csv"
    dated_csv = config.tables_dir / f"final_quant_model_1y_{date_label}.csv"
    save_dataframe(generic_csv, comparison_df)
    save_dataframe(dated_csv, comparison_df)

    best_rows = comparison_df.groupby("strategy_name", as_index=False).head(1).copy()
    best_rows = best_rows.sort_values(["excess_return_vs_spy", "sharpe_ratio", "max_drawdown"], ascending=[False, False, False])

    curves_df = pd.concat(best_weekly_by_strategy.values(), ignore_index=True) if best_weekly_by_strategy else pd.DataFrame()
    coverage_df = _sentiment_coverage(features, config.benchmark)

    equity_path = config.charts_dir / "final_quant_model_1y_equity_curves.png"
    equity_dated_path = config.charts_dir / f"final_quant_model_1y_equity_curves_{date_label}.png"
    drawdown_path = config.charts_dir / "final_quant_model_1y_drawdowns.png"
    drawdown_dated_path = config.charts_dir / f"final_quant_model_1y_drawdowns_{date_label}.png"
    comparison_path = config.charts_dir / "final_quant_model_1y_test_comparison.png"
    comparison_dated_path = config.charts_dir / f"final_quant_model_1y_test_comparison_{date_label}.png"
    coverage_path = config.charts_dir / "final_quant_model_1y_sentiment_coverage.png"
    coverage_dated_path = config.charts_dir / f"final_quant_model_1y_sentiment_coverage_{date_label}.png"

    if not curves_df.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        for strategy_name, group in curves_df.groupby("strategy_name"):
            ax.plot(group["date"], group["portfolio_value"], label=strategy_name)
        ax.set_title("Final Quant Model 1Y Equity Curves")
        ax.legend(fontsize=8, ncol=2)
        _save_plot(fig, equity_path)
        shutil.copy2(equity_path, equity_dated_path)

        fig, ax = plt.subplots(figsize=(12, 7))
        for strategy_name, group in curves_df.groupby("strategy_name"):
            drawdown = group["portfolio_value"] / group["portfolio_value"].cummax() - 1
            ax.plot(group["date"], drawdown, label=strategy_name)
        ax.set_title("Final Quant Model 1Y Drawdowns")
        ax.legend(fontsize=8, ncol=2)
        _save_plot(fig, drawdown_path)
        shutil.copy2(drawdown_path, drawdown_dated_path)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(best_rows["strategy_name"], best_rows["excess_return_vs_spy"])
    ax.set_title("Final Quant Model 1Y Excess Return vs SPY")
    ax.tick_params(axis="x", rotation=45)
    _save_plot(fig, comparison_path)
    shutil.copy2(comparison_path, comparison_dated_path)

    if not coverage_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(coverage_df["date"], coverage_df["coverage_pct_7d"], label="Universe sentiment coverage")
        ax.set_ylim(0, 1)
        ax.set_title("Final Quant Model 1Y Sentiment Coverage")
        ax.legend()
        _save_plot(fig, coverage_path)
        shutil.copy2(coverage_path, coverage_dated_path)

    def _best_for(strategy_name: str) -> pd.Series | None:
        subset = best_rows.loc[best_rows["strategy_name"] == strategy_name]
        return subset.iloc[0] if not subset.empty else None

    best_total = best_rows.sort_values("total_return", ascending=False).iloc[0]
    best_excess = best_rows.sort_values("excess_return_vs_spy", ascending=False).iloc[0]
    best_sharpe = best_rows.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_drawdown = best_rows.sort_values("max_drawdown", ascending=False).iloc[0]
    non_snapshot = best_rows.loc[best_rows["analyst_data_mode"] != "snapshot_current"]
    best_non_snapshot = non_snapshot.iloc[0] if not non_snapshot.empty else None
    best_spy = _best_for("SPY")
    best_snapshot = _best_for("analyst_snapshot_model")
    best_final = _best_for("final_quant_model_1y")
    best_final_no_sentiment = _best_for("final_quant_model_1y_no_sentiment")
    best_sector_capped = _best_for("final_quant_model_1y_sector_capped")
    best_strict = _best_for("strict_checklist_model")

    lines = [
        "# Final Quant Model 1Y Report",
        "",
        f"- Analysis window: {start_date} to {end_date}",
        "- Evaluation mode: short-sample evaluation",
        f"- Historical rating-count data available: {has_historical_ratings}",
        f"- Sentiment data available: {has_sentiment}",
        "",
        "## Model Selection Summary",
        f"- Best model by total return: {best_total['strategy_name']} ({best_total['total_return']:.2%})",
        f"- Best model by excess return vs SPY: {best_excess['strategy_name']} ({best_excess['excess_return_vs_spy']:.2%})",
        f"- Best model by Sharpe: {best_sharpe['strategy_name']} ({best_sharpe['sharpe_ratio']:.2f})",
        f"- Best model by max drawdown: {best_drawdown['strategy_name']} ({best_drawdown['max_drawdown']:.2%})",
        (
            f"- Best model that does NOT use snapshot analyst data: "
            f"{best_non_snapshot['strategy_name']} ({best_non_snapshot['excess_return_vs_spy']:.2%} excess)"
            if best_non_snapshot is not None
            else "- Best model that does NOT use snapshot analyst data: none available."
        ),
        f"- Whether any model beat SPY: {'Yes' if (best_excess['strategy_name'] != 'SPY' and best_excess['excess_return_vs_spy'] > 0) else 'No'}",
        (
            f"- Whether any model beat analyst_snapshot_model: "
            f"{'Yes' if best_snapshot is not None and (best_excess['excess_return_vs_spy'] > best_snapshot['excess_return_vs_spy']) else 'No'}"
        ),
        (
            f"- Whether sentiment improved returns: "
            f"{'Yes' if best_final is not None and best_final_no_sentiment is not None and best_final['excess_return_vs_spy'] > best_final_no_sentiment['excess_return_vs_spy'] else 'No or inconclusive'}"
        ),
        (
            f"- Whether sentiment improved drawdown: "
            f"{'Yes' if best_final is not None and best_final_no_sentiment is not None and best_final['max_drawdown'] > best_final_no_sentiment['max_drawdown'] else 'No or inconclusive'}"
        ),
        (
            f"- Whether strict checklist helped or hurt: "
            f"{'Helped' if best_strict is not None and _best_for('full_model') is not None and best_strict['excess_return_vs_spy'] > _best_for('full_model')['excess_return_vs_spy'] else 'Hurt or inconclusive'}"
        ),
        (
            f"- Whether sector caps helped or hurt: "
            f"{'Helped' if best_sector_capped is not None and best_final is not None and best_sector_capped['excess_return_vs_spy'] > best_final['excess_return_vs_spy'] else 'Hurt or inconclusive'}"
        ),
    ]

    if strategy_analyst_data_mode(best_excess["strategy_name"]) == "snapshot_current":
        lines.extend(
            [
                "",
                "The best-performing model uses snapshot analyst target data and should be treated as exploratory, not a valid historical analyst backtest.",
            ]
        )

    lines.extend(
        [
            "",
            "## Best Rows",
            "",
            _dataframe_to_markdown(
                best_rows[
                    [
                        "strategy_name",
                        "analyst_data_mode",
                        "holding_period_days",
                        "top_n",
                        "total_return",
                        "excess_return_vs_spy",
                        "sharpe_ratio",
                        "max_drawdown",
                        "average_turnover",
                        "average_holdings",
                        "number_of_rebalance_periods",
                    ]
                ].head(12).round(4)
            ),
            "",
        ]
    )

    if skipped_models:
        lines.extend(["## Skipped Models", *[f"- {item}" for item in sorted(set(skipped_models))], ""])

    lines.extend(
        [
            "## Caveats",
            f"- {ANALYST_SNAPSHOT_CAVEAT}",
            f"- {SENTIMENT_CAVEAT}",
            f"- {BACKTEST_CAVEAT}",
            f"- {ONE_YEAR_CAVEAT}",
        ]
    )

    report_text = "\n".join(lines) + "\n"
    generic_report = config.reports_dir / "final_quant_model_1y_report.md"
    dated_report = config.reports_dir / f"final_quant_model_1y_{date_label}.md"
    generic_report.write_text(report_text, encoding="utf-8")
    dated_report.write_text(report_text, encoding="utf-8")

    print(f"Saved comparison: {generic_csv}")
    print(f"Saved dated comparison: {dated_csv}")
    print(f"Saved report: {generic_report}")
    print(f"Saved dated report: {dated_report}")


if __name__ == "__main__":
    main()
