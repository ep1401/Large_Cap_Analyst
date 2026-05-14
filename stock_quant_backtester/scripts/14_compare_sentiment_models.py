from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.utils import load_dataframe, save_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven results currently use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out["date"] >= start]
    if end is not None:
        out = out.loc[out["date"] <= end]
    return out


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def _build_spy_returns(features: pd.DataFrame, benchmark: str, holding_period_days: int, initial_capital: float) -> pd.DataFrame:
    future_map = {
        5: ("future_5d_spy_return",),
        21: ("future_21d_spy_return",),
        63: ("future_63d_spy_return",),
    }
    future_spy_return_column = future_map[holding_period_days][0]
    rebalance_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=benchmark)
    spy_value = initial_capital
    rows: list[dict] = []
    for date in rebalance_dates:
        spy_return = float(
            features.loc[(features["ticker"] == benchmark) & (features["date"] == date), future_spy_return_column].iloc[0]
        )
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": "SPY",
                "holding_period_days": holding_period_days,
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


def _metrics_row(strategy_name: str, weekly: pd.DataFrame, *, holding_period_days: int, top_n: int, require_positive_sentiment: bool, avoid_strong_negative_news: bool, min_article_count_7d: int) -> dict:
    full = calculate_performance_metrics(weekly, holding_period_days=holding_period_days)
    dev = calculate_performance_metrics(_slice_period(weekly, end=DEV_END), holding_period_days=holding_period_days)
    test = calculate_performance_metrics(_slice_period(weekly, start=TEST_START), holding_period_days=holding_period_days)
    return {
        "strategy_name": strategy_name,
        "holding_period_days": holding_period_days,
        "top_n": top_n,
        "require_positive_sentiment": require_positive_sentiment,
        "avoid_strong_negative_news": avoid_strong_negative_news,
        "min_article_count_7d": min_article_count_7d,
        "full_period_total_return": full["total_return"],
        "development_period_total_return": dev["total_return"],
        "test_period_total_return": test["total_return"],
        "full_period_excess_return_vs_spy": full["excess_total_return"],
        "test_period_excess_return_vs_spy": test["excess_total_return"],
        "annualized_return": full["annualized_return"],
        "annualized_volatility": full["annualized_volatility"],
        "sharpe_ratio": full["sharpe_ratio"],
        "test_sharpe_ratio": test["sharpe_ratio"],
        "max_drawdown": full["max_drawdown"],
        "average_holdings": full["average_selected_count"],
        "average_turnover": full["average_turnover"],
        "number_of_rebalance_periods": full["number_of_rebalance_periods"],
    }


def _build_sentiment_diagnostics(features: pd.DataFrame, selected_holdings: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    selected_daily = (
        selected_holdings.groupby("date")
        .agg(
            selected_avg_news_sentiment_7d=("news_sentiment_7d", "mean"),
            selected_negative_news_ratio_7d=("negative_news_ratio_7d", "mean"),
            selected_positive_sentiment_count=("news_sentiment_7d", lambda s: int((s > 0).sum())),
            selected_holdings_count=("ticker", "size"),
        )
        .reset_index()
    )

    rows = []
    for date, day in features.loc[features["ticker"] != benchmark].groupby("date"):
        total_candidates = len(day)
        candidates_with_news_7d = int((day["article_count_7d"] >= 1).sum())
        candidates_with_positive_sentiment_7d = int((day["news_sentiment_7d"] > 0).sum())
        candidates_with_negative_sentiment_7d = int((day["news_sentiment_7d"] < 0).sum())
        candidates_with_strong_negative_news = int(day["strong_negative_news_flag"].fillna(False).sum())
        rows.append(
            {
                "date": pd.to_datetime(date),
                "total_candidates": total_candidates,
                "candidates_with_news_7d": candidates_with_news_7d,
                "candidates_with_positive_sentiment_7d": candidates_with_positive_sentiment_7d,
                "candidates_with_negative_sentiment_7d": candidates_with_negative_sentiment_7d,
                "candidates_with_strong_negative_news": candidates_with_strong_negative_news,
                "average_news_sentiment_7d": float(day["news_sentiment_7d"].mean()),
                "median_news_sentiment_7d": float(day["news_sentiment_7d"].median()),
                "coverage_pct_7d": float(candidates_with_news_7d / total_candidates) if total_candidates else 0.0,
                "avg_article_count_7d": float(day["article_count_7d"].mean()),
            }
        )
    diagnostics_df = pd.DataFrame(rows).merge(selected_daily, on="date", how="left")
    diagnostics_df["selected_avg_news_sentiment_7d"] = diagnostics_df["selected_avg_news_sentiment_7d"].fillna(0.0)
    diagnostics_df["selected_negative_news_ratio_7d"] = diagnostics_df["selected_negative_news_ratio_7d"].fillna(0.0)
    diagnostics_df["selected_holdings_count"] = diagnostics_df["selected_holdings_count"].fillna(0).astype(int)
    diagnostics_df["selected_positive_sentiment_count"] = diagnostics_df["selected_positive_sentiment_count"].fillna(0).astype(int)
    diagnostics_df["selected_positive_sentiment_pct"] = (
        diagnostics_df["selected_positive_sentiment_count"] / diagnostics_df["selected_holdings_count"].replace(0, pd.NA)
    ).fillna(0.0)
    return diagnostics_df.sort_values("date").reset_index(drop=True)


def _answer_questions(comparison_df: pd.DataFrame) -> list[str]:
    test_df = comparison_df.sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    )
    latest = {
        row["strategy_name"]: row
        for _, row in test_df.groupby("strategy_name", as_index=False).head(1).iterrows()
    }

    def beat(a: str, b: str) -> str:
        if a not in latest or b not in latest:
            return "Not enough data to answer."
        delta = latest[a]["test_period_excess_return_vs_spy"] - latest[b]["test_period_excess_return_vs_spy"]
        return "Yes." if delta > 0 else "No."

    best_no_sent = test_df.loc[
        test_df["strategy_name"].isin(["full_model", "strict_checklist_model", "analyst_only", "technical_only", "technical_momentum_model"])
    ].iloc[0]
    best_sent = test_df.loc[test_df["strategy_name"].isin([
        "full_model_with_sentiment",
        "strict_checklist_with_sentiment",
        "sentiment_only",
        "analyst_sentiment_model",
        "technical_sentiment_model",
    ])].iloc[0]

    return [
        f"- Does sentiment improve full_model? {beat('full_model_with_sentiment', 'full_model')}",
        f"- Does sentiment improve strict_checklist_model? {beat('strict_checklist_with_sentiment', 'strict_checklist_model')}",
        (
            "- Does avoiding strong negative news improve drawdown? "
            + (
                "Yes."
                if "strict_checklist_with_sentiment" in latest
                and (
                    test_df.loc[
                        (test_df["strategy_name"] == "strict_checklist_with_sentiment")
                        & (test_df["avoid_strong_negative_news"] == True),
                        "max_drawdown",
                    ].max()
                    > test_df.loc[
                        (test_df["strategy_name"] == "strict_checklist_with_sentiment")
                        & (test_df["avoid_strong_negative_news"] == False),
                        "max_drawdown",
                    ].max()
                )
                else "No or inconclusive."
            )
        ),
        (
            "- Does requiring positive sentiment improve or hurt diversification? "
            + (
                "It appears to hurt diversification."
                if "strict_checklist_with_sentiment" in latest
                and test_df.loc[
                    (test_df["strategy_name"] == "strict_checklist_with_sentiment")
                    & (test_df["require_positive_sentiment"] == True),
                    "average_holdings",
                ].mean()
                < test_df.loc[
                    (test_df["strategy_name"] == "strict_checklist_with_sentiment")
                    & (test_df["require_positive_sentiment"] == False),
                    "average_holdings",
                ].mean()
                else "It does not appear to hurt diversification materially."
            )
        ),
        f"- Does any sentiment model beat SPY on the test period? {'Yes.' if best_sent['test_period_excess_return_vs_spy'] > 0 else 'No.'}",
        f"- Does any sentiment model beat the best no-sentiment model on the test period? {'Yes.' if best_sent['test_period_excess_return_vs_spy'] > best_no_sent['test_period_excess_return_vs_spy'] else 'No.'}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    if args.start_date:
        features = features.loc[features["date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        features = features.loc[features["date"] < pd.Timestamp(args.end_date)].copy()
    if "sentiment_data_mode" not in features.columns or features["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all():
        raise SystemExit(
            "The current feature panel does not include built news sentiment data. "
            "Run scripts/12_fetch_alpha_vantage_news.py, scripts/13_build_news_sentiment.py, and scripts/04_build_features.py first."
        )
    from src.plots import create_sentiment_plots
    output_suffix = args.output_suffix.strip()
    suffix = f"_{output_suffix}" if output_suffix else ""

    strategy_specs = [
        {"strategy_name": "full_model", "use_analyst_filters": True},
        {"strategy_name": "full_model_with_sentiment", "use_analyst_filters": True},
        {"strategy_name": "strict_checklist_model", "use_analyst_filters": True},
        {"strategy_name": "strict_checklist_with_sentiment", "use_analyst_filters": True},
        {"strategy_name": "sentiment_only", "use_analyst_filters": False},
        {"strategy_name": "analyst_only", "use_analyst_filters": True},
        {"strategy_name": "analyst_sentiment_model", "use_analyst_filters": True},
        {"strategy_name": "technical_only", "use_analyst_filters": False},
        {"strategy_name": "technical_sentiment_model", "use_analyst_filters": False},
    ]
    if "technical_momentum_model" in {"technical_momentum_model"}:
        strategy_specs.append({"strategy_name": "technical_momentum_model", "use_analyst_filters": False})

    comparison_rows: list[dict] = []
    curves_rows: list[pd.DataFrame] = []
    selected_for_diagnostics = pd.DataFrame()

    for holding_period_days in [5, 21, 63]:
        spy_weekly = _build_spy_returns(features, config.benchmark, holding_period_days, config.initial_capital)
        comparison_rows.append(
            _metrics_row(
                "SPY",
                spy_weekly,
                holding_period_days=holding_period_days,
                top_n=1,
                require_positive_sentiment=False,
                avoid_strong_negative_news=False,
                min_article_count_7d=0,
            )
        )

        for top_n in [10, 20]:
            for spec in strategy_specs:
                strategy_name = spec["strategy_name"]
                if strategy_name == "strict_checklist_with_sentiment":
                    sentiment_grids = [
                        (require_positive_sentiment, avoid_strong_negative_news, min_article_count_7d)
                        for require_positive_sentiment in [False, True]
                        for avoid_strong_negative_news in [False, True]
                        for min_article_count_7d in [0, 1, 3]
                    ]
                else:
                    sentiment_grids = [(False, False, 0)]

                for require_positive_sentiment, avoid_strong_negative_news, min_article_count_7d in sentiment_grids:
                    weekly, holdings, _ = run_weekly_backtest(
                        features=features,
                        holding_period_days=holding_period_days,
                        benchmark=config.benchmark,
                        top_n=top_n,
                        initial_capital=config.initial_capital,
                        transaction_cost_bps=config.transaction_cost_bps,
                        use_regime_filter=False,
                        regime_exposure=0.0,
                        use_analyst_filters=spec["use_analyst_filters"],
                        analyst_count_threshold=10,
                        min_avg_dollar_volume=20_000_000,
                        strategy_name=strategy_name,
                        require_positive_sentiment=require_positive_sentiment,
                        avoid_strong_negative_news=avoid_strong_negative_news,
                        min_article_count_7d=min_article_count_7d,
                    )
                    comparison_rows.append(
                        _metrics_row(
                            strategy_name,
                            weekly,
                            holding_period_days=holding_period_days,
                            top_n=top_n,
                            require_positive_sentiment=require_positive_sentiment,
                            avoid_strong_negative_news=avoid_strong_negative_news,
                            min_article_count_7d=min_article_count_7d,
                        )
                    )
                    if holding_period_days == 21 and top_n == 10 and strategy_name in {
                        "SPY",
                        "full_model",
                        "full_model_with_sentiment",
                        "strict_checklist_model",
                        "strict_checklist_with_sentiment",
                        "sentiment_only",
                        "analyst_sentiment_model",
                        "technical_sentiment_model",
                    }:
                        curves_rows.append(weekly[["date", "strategy_name", "portfolio_value"]].copy())
                    if (
                        holding_period_days == 21
                        and top_n == 10
                        and strategy_name == "full_model_with_sentiment"
                        and not require_positive_sentiment
                        and not avoid_strong_negative_news
                        and min_article_count_7d == 0
                    ):
                        selected_for_diagnostics = holdings.copy()

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "sentiment_model_comparison.csv", comparison_df)
    if suffix:
        save_dataframe(config.tables_dir / f"sentiment_model_comparison{suffix}.csv", comparison_df)
    else:
        save_dataframe(
            config.tables_dir / f"sentiment_model_comparison_{config.sentiment_window_label}.csv",
            comparison_df,
        )

    diagnostics_df = _build_sentiment_diagnostics(features, selected_for_diagnostics, config.benchmark)
    save_dataframe(config.tables_dir / "sentiment_diagnostics.csv", diagnostics_df)
    if suffix:
        save_dataframe(config.tables_dir / f"sentiment_diagnostics{suffix}.csv", diagnostics_df)

    curves_df = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame(columns=["date", "strategy_name", "portfolio_value"])
    create_sentiment_plots(diagnostics_df, comparison_df, curves_df, config.charts_dir)
    if suffix and (config.charts_dir / "sentiment_strategy_equity_curves.png").exists():
        source = config.charts_dir / "sentiment_strategy_equity_curves.png"
        shutil.copyfile(source, config.charts_dir / f"sentiment_strategy_equity_curves{suffix}.png")

    best_sentiment = comparison_df.loc[
        comparison_df["strategy_name"].isin(
            [
                "full_model_with_sentiment",
                "strict_checklist_with_sentiment",
                "sentiment_only",
                "analyst_sentiment_model",
                "technical_sentiment_model",
            ]
        )
    ].head(15)

    lines = [
        "# Sentiment Model Comparison",
        "",
        f"- Start date: {args.start_date or config.start_date}",
        f"- End date: {args.end_date or config.end_date}",
        f"- Benchmark: {config.benchmark}",
        f"- {IMPORTANT_CAVEAT}",
        "",
        "## Sentiment Coverage Diagnostics",
        f"- Percent of universe with at least one article in prior 7 days: {diagnostics_df['coverage_pct_7d'].mean():.2%}",
        f"- Average article_count_7d: {diagnostics_df['avg_article_count_7d'].mean():.2f}",
        f"- Percent of selected holdings with positive sentiment: {diagnostics_df['selected_positive_sentiment_pct'].mean():.2%}",
        (
            f"- Whether sentiment filters are too restrictive: "
            f"{(diagnostics_df['coverage_pct_7d'].mean() < 0.15) or (comparison_df['average_holdings'].mean() < 3)}"
        ),
        "",
        "## Test Period Leaders",
        "",
        _dataframe_to_markdown(best_sentiment.round(6)),
        "",
        "## Final Answers",
        *_answer_questions(comparison_df),
        "",
        "## Caveats",
        "- News sentiment is based on available Alpha Vantage news coverage and locally generated model scores. Missing articles, source coverage differences, publication timing, and model classification errors may affect historical accuracy.",
        f"- {IMPORTANT_CAVEAT}",
    ]
    report_path = config.reports_dir / "sentiment_model_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if suffix:
        suffixed_report = config.reports_dir / f"sentiment_model_comparison{suffix}.md"
        suffixed_report.write_text("\n".join(lines), encoding="utf-8")
    else:
        dated_report = config.reports_dir / f"sentiment_model_comparison_{config.sentiment_window_label}.md"
        dated_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved sentiment comparison report to {report_path}")


if __name__ == "__main__":
    main()
