from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_condition_based_backtest, run_weekly_backtest, select_rebalance_dates
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
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *rows])


def _metrics_by_period(returns_df: pd.DataFrame, holding_period_days: int) -> dict[str, dict]:
    periods = {
        "full": returns_df,
        "dev": _slice_period(returns_df, end=DEV_END),
        "test": _slice_period(returns_df, start=TEST_START),
    }
    return {
        label: calculate_performance_metrics(period_df, holding_period_days=holding_period_days)
        for label, period_df in periods.items()
        if not period_df.empty
    }


def _assemble_comparison_row(name: str, returns_df: pd.DataFrame, holding_period_days: int, number_of_trades: int = 0, average_holding_days: float | None = None) -> dict:
    metrics = _metrics_by_period(returns_df, holding_period_days)
    full = metrics["full"]
    dev = metrics.get("dev", full)
    test = metrics.get("test", full)
    return {
        "strategy_name": name,
        "holding_period_days": holding_period_days,
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
        "win_rate": full["win_rate"],
        "periods_beating_spy": full["weeks_beating_spy"],
        "average_turnover": full["average_turnover"],
        "average_holdings": full["average_selected_count"],
        "number_of_trades": number_of_trades,
        "average_holding_days": average_holding_days,
    }


def _run_custom_rank_backtest(
    features: pd.DataFrame,
    benchmark: str,
    holding_period_days: int,
    score_column: str | None = None,
    top_n: int | None = None,
    average_all: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    future_return_col, future_spy_return_col, _ = {
        5: ("future_5d_return", "future_5d_spy_return", "future_5d_excess_return"),
        21: ("future_21d_return", "future_21d_spy_return", "future_21d_excess_return"),
        63: ("future_63d_return", "future_63d_spy_return", "future_63d_excess_return"),
    }[holding_period_days]
    rebalance_dates = select_rebalance_dates(features, holding_period_days, benchmark)
    value = 10000.0
    spy_value = 10000.0
    rows = []
    rng = np.random.default_rng(random_seed)

    for date in rebalance_dates:
        day = features.loc[(features["date"] == date) & (features["ticker"] != benchmark)].copy()
        day = day.loc[day[future_return_col].notna()].copy()
        if average_all:
            selected = day
        elif score_column == "__random__":
            if len(day) > 0:
                sample_n = min(top_n or 10, len(day))
                selected = day.iloc[rng.choice(len(day), size=sample_n, replace=False)]
            else:
                selected = day
        else:
            selected = day.sort_values(score_column, ascending=False).head(top_n or 10)

        selected_count = len(selected)
        gross_return = float(selected[future_return_col].mean()) if selected_count else 0.0
        spy_slice = features.loc[(features["date"] == date) & (features["ticker"] == benchmark), future_spy_return_col]
        spy_return = float(spy_slice.iloc[0]) if not spy_slice.empty else 0.0
        value *= 1 + gross_return
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": score_column or "equal_weight_universe",
                "holding_period_days": holding_period_days,
                "selected_count": selected_count,
                "qualified_count": selected_count,
                "gross_return": gross_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": gross_return,
                "spy_return": spy_return,
                "excess_return": gross_return - spy_return,
                "portfolio_value": value,
                "spy_value": spy_value,
                "exposure": 1.0 if selected_count else 0.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _build_benchmark_buy_hold(features: pd.DataFrame, ticker: str, benchmark: str, holding_period_days: int = 5) -> pd.DataFrame:
    future_return_col, future_spy_return_col, _ = {
        5: ("future_5d_return", "future_5d_spy_return", "future_5d_excess_return"),
        21: ("future_21d_return", "future_21d_spy_return", "future_21d_excess_return"),
        63: ("future_63d_return", "future_63d_spy_return", "future_63d_excess_return"),
    }[holding_period_days]
    rebalance_dates = select_rebalance_dates(features, holding_period_days, benchmark)
    value = 10000.0
    spy_value = 10000.0
    rows = []
    for date in rebalance_dates:
        ticker_slice = features.loc[(features["date"] == date) & (features["ticker"] == ticker), future_return_col]
        if ticker_slice.empty:
            continue
        period_return = float(ticker_slice.iloc[0])
        spy_return = float(features.loc[(features["date"] == date) & (features["ticker"] == benchmark), future_spy_return_col].iloc[0])
        value *= 1 + period_return
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": f"{ticker}_buy_and_hold",
                "holding_period_days": holding_period_days,
                "selected_count": 1,
                "qualified_count": 1,
                "gross_return": period_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": period_return,
                "spy_return": spy_return,
                "excess_return": period_return - spy_return,
                "portfolio_value": value,
                "spy_value": spy_value,
                "exposure": 1.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    rows: list[dict] = []

    strategy_runs = [
        ("analyst_only_fixed_21", dict(strategy_name="analyst_only", holding_period_days=21)),
        ("analyst_only_fixed_63", dict(strategy_name="analyst_only", holding_period_days=63)),
        ("technical_only_fixed_21", dict(strategy_name="technical_only", holding_period_days=21)),
        ("technical_only_fixed_63", dict(strategy_name="technical_only", holding_period_days=63)),
        ("technical_momentum_model_fixed_21", dict(strategy_name="technical_momentum_model", holding_period_days=21, use_analyst_filters=False)),
        ("technical_momentum_model_fixed_63", dict(strategy_name="technical_momentum_model", holding_period_days=63, use_analyst_filters=False)),
        ("full_model_fixed_21", dict(strategy_name="full_model", holding_period_days=21)),
        ("full_model_fixed_63", dict(strategy_name="full_model", holding_period_days=63)),
        ("strict_checklist_model_fixed_21", dict(strategy_name="strict_checklist_model", holding_period_days=21, require_positive_revision_7d=False, resistance_window=63, resistance_distance_threshold=0.03)),
        ("strict_checklist_model_fixed_63", dict(strategy_name="strict_checklist_model", holding_period_days=63, require_positive_revision_7d=False, resistance_window=63, resistance_distance_threshold=0.03)),
    ]

    for label, kwargs in strategy_runs:
        weekly, _, _ = run_weekly_backtest(
            features=features,
            benchmark=config.benchmark,
            top_n=config.top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            analyst_count_threshold=10,
            min_avg_dollar_volume=20_000_000,
            use_regime_filter=False,
            regime_exposure=0.0,
            **kwargs,
        )
        rows.append(_assemble_comparison_row(label, weekly, holding_period_days=kwargs["holding_period_days"]))

    for label, strategy_name in [("condition_based_full_model", "full_model"), ("condition_based_strict_checklist_model", "strict_checklist_model")]:
        returns_df, holdings_df, trades_df = run_condition_based_backtest(
            features=features,
            strategy_name=strategy_name,
            top_n=config.top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            benchmark=config.benchmark,
            analyst_count_threshold=10,
            min_avg_dollar_volume=20_000_000,
            use_regime_filter=False,
            regime_exposure=0.0,
        )
        avg_holding_days = float(holdings_df["holding_days"].mean()) if not holdings_df.empty else None
        rows.append(
            _assemble_comparison_row(
                label,
                returns_df,
                holding_period_days=5,
                number_of_trades=len(trades_df),
                average_holding_days=avg_holding_days,
            )
        )

    spy_df = _build_benchmark_buy_hold(features, ticker=config.benchmark, benchmark=config.benchmark, holding_period_days=5)
    rows.append(_assemble_comparison_row("SPY", spy_df, holding_period_days=5))
    if "QQQ" in set(features["ticker"]):
        qqq_df = _build_benchmark_buy_hold(features, ticker="QQQ", benchmark=config.benchmark, holding_period_days=5)
        rows.append(_assemble_comparison_row("QQQ", qqq_df, holding_period_days=5))

    eq_df = _run_custom_rank_backtest(features, config.benchmark, holding_period_days=5, average_all=True)
    rows.append(_assemble_comparison_row("equal_weight_universe", eq_df, holding_period_days=5))
    mom21_df = _run_custom_rank_backtest(features, config.benchmark, holding_period_days=21, score_column="return_21d", top_n=10)
    rows.append(_assemble_comparison_row("top_10_momentum_21d", mom21_df, holding_period_days=21))
    mom63_df = _run_custom_rank_backtest(features, config.benchmark, holding_period_days=63, score_column="return_63d", top_n=10)
    rows.append(_assemble_comparison_row("top_10_momentum_63d", mom63_df, holding_period_days=63))

    random_dfs = []
    for seed in range(100):
        random_dfs.append(_run_custom_rank_backtest(features, config.benchmark, holding_period_days=5, score_column="__random__", top_n=10, random_seed=seed))
    random_returns = pd.concat(random_dfs).groupby("date", as_index=False).mean(numeric_only=True)
    random_returns["strategy_name"] = "random_10_universe"
    rows.append(_assemble_comparison_row("random_10_universe", random_returns, holding_period_days=5))

    comparison = pd.DataFrame(rows).sort_values(
        by=["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    )
    save_dataframe(config.tables_dir / "model_improvement_comparison.csv", comparison)

    lines = [
        "# Model Improvement Comparison",
        "",
        f"- {IMPORTANT_CAVEAT}",
        "",
        _dataframe_to_markdown(comparison.round(6)),
    ]
    report_path = config.reports_dir / "model_improvement_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved model improvement comparison to {config.tables_dir / 'model_improvement_comparison.csv'}")
    print(f"Saved model improvement report to {report_path}")


if __name__ == "__main__":
    main()
