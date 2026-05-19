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
    CustomStrategyDefinition,
    build_final_quant_5d_definition,
    dataframe_to_markdown,
    fmt_pct,
    get_best_5d_config,
    load_features,
    run_benchmark_buy_hold,
    run_custom_weekly_backtest,
    summarize_backtest,
)
from src.utils import save_dataframe


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((values - values.mean()) / std).clip(-3, 3).fillna(0.0)


def _technical_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score"] = (
        0.25 * _zscore(df["relative_strength_21d"])
        + 0.20 * _zscore(df["relative_strength_63d"])
        + 0.15 * _zscore(-df["distance_to_63d_high"])
        + 0.10 * _zscore(df["above_sma_50"])
        + 0.10 * _zscore(df["above_sma_200"])
        - 0.05 * _zscore(df["volatility_21d"])
        + 0.05 * df["breakout_63d"].fillna(False).astype(float)
    )
    return scored


def _single_factor_definition(name: str, display_name: str, column: str) -> CustomStrategyDefinition:
    return CustomStrategyDefinition(
        name=name,
        display_name=display_name,
        score_builder=lambda df, factor=column: df.assign(score=_zscore(df[factor])),
    )


def _technical_definition() -> CustomStrategyDefinition:
    return CustomStrategyDefinition(
        name="top_10_technical_momentum",
        display_name="Top 10 Technical Momentum",
        score_builder=_technical_momentum_score,
    )


def _run_random_average(
    features: pd.DataFrame,
    holding_period_days: int,
    benchmark: str,
    transaction_cost_bps: float,
    min_avg_dollar_volume: float,
) -> pd.DataFrame:
    seed_summaries: list[dict[str, float]] = []
    for seed in range(100):
        definition = CustomStrategyDefinition(
            name=f"random_10_universe_seed_{seed}",
            display_name=f"Random 10 Universe Seed {seed}",
            score_builder=lambda df, local_seed=seed: df.assign(
                score=pd.Series(np.random.RandomState(local_seed).permutation(len(df)), index=df.index)
            ),
        )
        weekly, _, _ = run_custom_weekly_backtest(
            features=features,
            definition=definition,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
            top_n=10,
            transaction_cost_bps=transaction_cost_bps,
            min_avg_dollar_volume=min_avg_dollar_volume,
        )
        seed_summaries.append(summarize_backtest(weekly, holding_period_days, f"seed_{seed}"))
    average = pd.DataFrame(seed_summaries).mean(numeric_only=True).to_dict()
    average["label"] = "random_10_universe"
    average["holding_period_days"] = holding_period_days
    return pd.DataFrame([average])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)
    qqq_available = "QQQ" in set(features["ticker"].dropna().unique())

    baseline_definitions = [
        _single_factor_definition("top_10_relative_strength_21d", "Top 10 Relative Strength 21D", "relative_strength_21d"),
        _single_factor_definition("top_10_relative_strength_63d", "Top 10 Relative Strength 63D", "relative_strength_63d"),
        _single_factor_definition("top_10_historical_rating_score", "Top 10 Historical Rating Score", "historical_rating_score"),
        _single_factor_definition("top_10_net_upgrade_score_30d", "Top 10 Net Upgrade Score 30D", "net_upgrade_score_30d"),
        _single_factor_definition("top_10_sentiment_7d", "Top 10 Sentiment 7D", "relevance_weighted_sentiment_7d"),
        _technical_definition(),
    ]

    result_rows: list[dict[str, object]] = []
    for holding_period_days in [5, 21]:
        if holding_period_days == 5:
            model_definition = build_final_quant_5d_definition()
            model_weekly, _, _ = run_custom_weekly_backtest(
                features=features,
                definition=model_definition,
                holding_period_days=5,
                benchmark=config.benchmark,
                top_n=int(best_config["top_n"]),
                transaction_cost_bps=float(best_config["total_cost_bps"]),
                min_avg_dollar_volume=config.min_avg_dollar_volume,
                max_names_per_sector=best_config["max_names_per_sector"],
                position_sizing=str(best_config["position_sizing"]),
                max_single_name_weight=float(best_config["max_single_name_weight"]),
            )
            model_label = "final_quant_5d_no_snapshot_no_sma_filter"
            model_display = "Final Quant 5D - No SMA Filter"
        else:
            model_weekly, _, _ = run_weekly_backtest(
                features=features,
                holding_period_days=21,
                benchmark=config.benchmark,
                top_n=10,
                initial_capital=config.initial_capital,
                transaction_cost_bps=config.transaction_cost_bps,
                use_regime_filter=False,
                use_analyst_filters=False,
                analyst_count_threshold=config.analyst_count_threshold,
                min_avg_dollar_volume=config.min_avg_dollar_volume,
                strategy_name="final_quant_21d_no_snapshot_sector_capped",
                max_names_per_sector=4,
                position_sizing="equal_weight",
                min_historical_rating_count=5,
            )
            model_label = "final_quant_21d_no_snapshot_sector_capped"
            model_display = "Final Quant 21D - Sector Capped"

        model_summary = summarize_backtest(model_weekly, holding_period_days, model_label)
        model_summary["display_name"] = model_display
        model_summary["category"] = "final_model"
        result_rows.append(model_summary)

        spy_summary = summarize_backtest(
            run_benchmark_buy_hold(features, holding_period_days, config.benchmark, config.benchmark),
            holding_period_days,
            "spy",
        )
        spy_summary["display_name"] = "SPY"
        spy_summary["category"] = "benchmark"
        result_rows.append(spy_summary)

        if qqq_available:
            qqq_summary = summarize_backtest(
                run_benchmark_buy_hold(features, holding_period_days, "QQQ", config.benchmark),
                holding_period_days,
                "qqq",
            )
            qqq_summary["display_name"] = "QQQ"
            qqq_summary["category"] = "benchmark"
            result_rows.append(qqq_summary)

        equal_weight_definition = CustomStrategyDefinition(
            name="equal_weight_universe",
            display_name="Equal Weight Universe",
            score_builder=lambda df: df.assign(score=0.0),
        )
        equal_weight_weekly, _, _ = run_custom_weekly_backtest(
            features=features,
            definition=equal_weight_definition,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            top_n=None,
            transaction_cost_bps=config.transaction_cost_bps,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
        )
        equal_weight_summary = summarize_backtest(equal_weight_weekly, holding_period_days, "equal_weight_universe")
        equal_weight_summary["display_name"] = "Equal Weight Universe"
        equal_weight_summary["category"] = "baseline"
        result_rows.append(equal_weight_summary)

        for definition in baseline_definitions:
            weekly, _, _ = run_custom_weekly_backtest(
                features=features,
                definition=definition,
                holding_period_days=holding_period_days,
                benchmark=config.benchmark,
                top_n=10,
                transaction_cost_bps=config.transaction_cost_bps,
                min_avg_dollar_volume=config.min_avg_dollar_volume,
            )
            summary = summarize_backtest(weekly, holding_period_days, definition.name)
            summary["display_name"] = definition.display_name
            summary["category"] = "baseline"
            result_rows.append(summary)

        random_summary = _run_random_average(
            features=features,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            transaction_cost_bps=config.transaction_cost_bps,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
        ).iloc[0].to_dict()
        random_summary["display_name"] = "Random 10 Universe"
        random_summary["category"] = "baseline"
        result_rows.append(random_summary)

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(
        ["holding_period_days", "full_period_excess_return_vs_spy", "sharpe_ratio"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "simple_baseline_comparison.csv", results_df)

    focus_5d = results_df.loc[results_df["holding_period_days"] == 5].copy()
    final_5d = focus_5d.loc[focus_5d["label"] == "final_quant_5d_no_snapshot_no_sma_filter"].iloc[0]
    best_momentum = focus_5d.loc[
        focus_5d["label"].isin(["top_10_relative_strength_21d", "top_10_relative_strength_63d", "top_10_technical_momentum"])
    ]["full_period_excess_return_vs_spy"].max()
    sentiment_only = float(focus_5d.loc[focus_5d["label"] == "top_10_sentiment_7d", "full_period_excess_return_vs_spy"].iloc[0])
    historical_only = float(focus_5d.loc[focus_5d["label"] == "top_10_historical_rating_score", "full_period_excess_return_vs_spy"].iloc[0])
    equal_weight = float(focus_5d.loc[focus_5d["label"] == "equal_weight_universe", "full_period_excess_return_vs_spy"].iloc[0])
    qqq_line = "- QQQ not available in the current universe/pricing panel."
    if qqq_available:
        qqq_excess = float(focus_5d.loc[focus_5d["label"] == "qqq", "full_period_excess_return_vs_spy"].iloc[0])
        qqq_line = (
            f"- Final 5D model beats QQQ on full-period excess vs SPY: "
            f"{final_5d['full_period_excess_return_vs_spy'] > qqq_excess} "
            f"({fmt_pct(final_5d['full_period_excess_return_vs_spy'])} vs {fmt_pct(qqq_excess)})."
        )

    report_view = results_df[
        [
            "holding_period_days",
            "label",
            "display_name",
            "category",
            "full_period_total_return",
            "full_period_excess_return_vs_spy",
            "sharpe_ratio",
            "max_drawdown",
            "average_turnover",
            "average_holdings",
            "windows_beating_spy",
        ]
    ].copy()
    for column in ["full_period_total_return", "full_period_excess_return_vs_spy", "sharpe_ratio", "max_drawdown", "average_turnover", "average_holdings"]:
        report_view[column] = report_view[column].round(4)

    report_lines = [
        "# Simple Baseline Comparison",
        "",
        "- 5-day comparison uses the current honest 5D model: `final_quant_5d_no_snapshot_no_sma_filter`.",
        "- 21-day comparison uses the current 21D no-snapshot sibling: `final_quant_21d_no_snapshot_sector_capped`.",
        "- Conclusions below are based on the full 2023-2025 window, with walk-forward columns included in the table.",
        "",
        "## Headline Answers",
        f"- Final 5D model beats the best simple momentum baseline on full-period excess vs SPY: {final_5d['full_period_excess_return_vs_spy'] > best_momentum} ({fmt_pct(final_5d['full_period_excess_return_vs_spy'])} vs {fmt_pct(best_momentum)}).",
        f"- Final 5D model beats sentiment-only on full-period excess vs SPY: {final_5d['full_period_excess_return_vs_spy'] > sentiment_only} ({fmt_pct(final_5d['full_period_excess_return_vs_spy'])} vs {fmt_pct(sentiment_only)}).",
        f"- Final 5D model beats historical-rating-only on full-period excess vs SPY: {final_5d['full_period_excess_return_vs_spy'] > historical_only} ({fmt_pct(final_5d['full_period_excess_return_vs_spy'])} vs {fmt_pct(historical_only)}).",
        f"- Final 5D model beats the equal-weight universe on full-period excess vs SPY: {final_5d['full_period_excess_return_vs_spy'] > equal_weight} ({fmt_pct(final_5d['full_period_excess_return_vs_spy'])} vs {fmt_pct(equal_weight)}).",
        qqq_line,
        "",
        "## Results",
        "",
        dataframe_to_markdown(report_view),
    ]
    (config.reports_dir / "simple_baseline_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved baseline comparison table to {config.tables_dir / 'simple_baseline_comparison.csv'}")
    print(f"Saved baseline comparison report to {config.reports_dir / 'simple_baseline_comparison.md'}")


if __name__ == "__main__":
    main()
