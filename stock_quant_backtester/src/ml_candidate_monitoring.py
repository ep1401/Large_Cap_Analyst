from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.recommended_strategy import (
    load_recommended_strategy_config,
    precompute_recommended_low_turnover_panels,
)
from src.research_models import load_ml_artifact, load_ml_research_candidate_config, precompute_research_panels
from src.utils import load_dataframe


INITIAL_CAPITAL = 10000.0
EXPECTED_RULE_BASED = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
SENTIMENT_FEATURES = [
    "relevance_weighted_sentiment_7d",
    "relevance_weighted_sentiment_30d",
    "sentiment_change_7d_vs_30d",
    "negative_news_ratio_7d",
    "market_sentiment_7d",
    "market_sentiment_30d",
    "market_negative_news_ratio_7d",
    "percent_tickers_positive_sentiment_7d",
    "percent_tickers_negative_sentiment_7d",
]
RATING_FEATURES = [
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "historical_rating_score_change_30d",
]
GRADE_EVENT_FEATURES = [
    "net_upgrade_score_30d",
    "downgrade_count_30d",
    "recent_downgrade_flag_30d",
]
TECHNICAL_FEATURES = [
    "relative_strength_21d",
    "relative_strength_63d",
    "volatility_21d",
    "beta_to_spy_63d",
    "distance_to_63d_high",
    "breakout_63d",
    "market_risk_score",
    "spy_return_21d",
    "spy_volatility_21d",
    "spy_drawdown_from_63d_high",
    "spy_above_sma_50",
    "spy_above_sma_200",
]


def ml_report_caveat_lines() -> list[str]:
    return [
        "This is a frozen ML research candidate.",
        "2026 data was not used for training, tuning, or model selection.",
        "Strict leakage timing audits passed, but ML can still overfit.",
        "Back-tested performance is hypothetical unless actually paper-tracked live.",
        "Snapshot analyst target fields are excluded.",
        "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "News sentiment depends on Alpha Vantage coverage and classification.",
        "ML models may overfit and require extended forward validation.",
        "This is research/paper trading only, not financial advice.",
    ]


def compute_drawdown(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric / numeric.cummax() - 1.0


def ensure_market_features(runtime: Config, features_forward: pd.DataFrame) -> pd.DataFrame:
    merged = features_forward.copy()
    sentiment_path = runtime.processed_dir / "market_sentiment_daily.csv"
    regime_path = runtime.processed_dir / "market_regime_daily.csv"

    if sentiment_path.exists():
        sentiment = load_dataframe(sentiment_path, parse_dates=["date"])
        sentiment_cols = [
            "market_sentiment_7d",
            "market_sentiment_30d",
            "market_sentiment_change_7d_vs_30d",
            "market_negative_news_ratio_7d",
            "percent_tickers_positive_sentiment_7d",
            "percent_tickers_negative_sentiment_7d",
            "sentiment_dispersion_7d",
        ]
        missing = [column for column in sentiment_cols if column not in merged.columns]
        if missing:
            merged = merged.merge(
                sentiment[["date", *[column for column in missing if column in sentiment.columns]]],
                on="date",
                how="left",
            )

    if regime_path.exists():
        regime = load_dataframe(regime_path, parse_dates=["date"])
        regime_cols = [
            "market_risk_score",
            "market_regime_label",
            "normalized_market_risk_score",
            "spy_return_21d",
            "spy_volatility_21d",
            "spy_drawdown_from_63d_high",
            "spy_above_sma_50",
            "spy_above_sma_200",
        ]
        missing = [column for column in regime_cols if column not in merged.columns]
        if missing:
            merged = merged.merge(
                regime[["date", *[column for column in missing if column in regime.columns]]],
                on="date",
                how="left",
            )

    if "spy_above_sma_50" not in merged.columns and "above_sma_50" in merged.columns:
        benchmark_rows = (
            merged.loc[merged["ticker"] == "SPY", ["date", "above_sma_50"]]
            .drop_duplicates(subset=["date"])
            .rename(columns={"above_sma_50": "spy_above_sma_50"})
        )
        merged = merged.merge(benchmark_rows, on="date", how="left")
    return merged


def lag_columns_by_trading_day(df: pd.DataFrame, columns: list[str], days: int) -> pd.DataFrame:
    shifted = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    present = [column for column in columns if column in shifted.columns]
    for column in present:
        shifted[column] = shifted.groupby("ticker")[column].shift(days)
    return shifted


def apply_variant_features(base_features: pd.DataFrame, variant_name: str, ml_feature_names: list[str] | None = None) -> pd.DataFrame:
    if variant_name == "normal_features":
        return base_features.copy()
    if variant_name == "sentiment_lag_1d":
        return lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES, 1)
    if variant_name == "sentiment_lag_2d":
        return lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES, 2)
    if variant_name == "ratings_lag_1d":
        return lag_columns_by_trading_day(base_features, RATING_FEATURES, 1)
    if variant_name == "grade_events_lag_1d":
        return lag_columns_by_trading_day(base_features, GRADE_EVENT_FEATURES, 1)
    if variant_name == "technical_lag_1d":
        return lag_columns_by_trading_day(base_features, TECHNICAL_FEATURES, 1)
    if variant_name == "all_non_price_alt_data_lag_1d":
        return lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES + RATING_FEATURES + GRADE_EVENT_FEATURES, 1)
    if variant_name == "all_features_lag_1d":
        if ml_feature_names is None:
            raise ValueError("ml_feature_names are required for all_features_lag_1d")
        return lag_columns_by_trading_day(base_features, list(ml_feature_names), 1)
    raise ValueError(f"Unsupported variant_name: {variant_name}")


def load_frozen_ml_context() -> tuple[Config, object, dict[str, object], pd.DataFrame]:
    runtime = Config.from_env()
    candidate = load_ml_research_candidate_config(runtime.project_root)
    artifact = load_ml_artifact(runtime.project_root / candidate.model_path)
    features_forward = load_dataframe(runtime.final_dir / "features_panel_2026_forward.csv", parse_dates=["date"])
    features_forward = ensure_market_features(runtime, features_forward)
    features_forward = features_forward.loc[features_forward["date"] >= pd.Timestamp(candidate.forward_window_start)].copy()
    return runtime, candidate, artifact, features_forward


def _simulate_ranked_panels(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    *,
    top_n: int,
    cost_bps: float,
    enter_rank: int,
    hold_rank: int,
    max_holding_days: int,
    rebalance_frequency_days: int,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    portfolio_value = INITIAL_CAPITAL
    spy_value = INITIAL_CAPITAL

    if len(panels) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for idx in range(len(panels) - 1):
        rebalance_date, panel, spy_price, _ = panels[idx]
        next_period_date = pd.Timestamp(panels[idx + 1][0])
        prior_weights = {ticker: float(meta.get("weight", 0.0)) for ticker, meta in holdings.items()}
        ranked = panel.sort_values("score", ascending=False).reset_index(drop=True).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_by_ticker = ranked.set_index("ticker", drop=False)
        next_panel_by_ticker = panels[idx + 1][3].set_index("ticker", drop=False)
        next_spy_price = float(panels[idx + 1][2])

        forced_sells: list[tuple[str, str]] = []
        discretionary_sells: list[tuple[str, int]] = []
        for ticker, meta in list(holdings.items()):
            if ticker not in ranked_by_ticker.index:
                forced_sells.append((ticker, "price_data_missing"))
                continue
            row = ranked_by_ticker.loc[ticker]
            rank = int(row["rank"])
            if bool(row["strong_negative_news_flag"]):
                forced_sells.append((ticker, "strong_negative_news_flag"))
            elif bool(row["recent_downgrade_flag_30d"]):
                forced_sells.append((ticker, "recent_downgrade_flag_30d"))
            elif int(meta["holding_days"]) >= max_holding_days:
                forced_sells.append((ticker, "max_holding_days"))
            elif rank > hold_rank:
                discretionary_sells.append((ticker, rank))

        for ticker, reason in forced_sells:
            action_rows.append({"date": rebalance_date, "period_end_date": next_period_date, "ticker": ticker, "action": "SELL", "reason": reason})
            holdings.pop(ticker, None)

        for ticker, rank in sorted(discretionary_sells, key=lambda item: item[1], reverse=True):
            action_rows.append({"date": rebalance_date, "period_end_date": next_period_date, "ticker": ticker, "action": "SELL", "reason": f"rank>{hold_rank}:{rank}"})
            holdings.pop(ticker, None)

        current_tickers = list(holdings.keys())
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_tickers
        ]
        open_slots = max(0, top_n - len(current_tickers))
        for ticker in desired_buys[:open_slots]:
            holdings[ticker] = {"holding_days": 0}
            action_rows.append({"date": rebalance_date, "period_end_date": next_period_date, "ticker": ticker, "action": "BUY", "reason": f"enter_rank<={enter_rank}"})

        selected_tickers = [ticker for ticker in ranked["ticker"].tolist() if ticker in holdings][:top_n]
        for ticker in list(holdings):
            if ticker not in selected_tickers:
                holdings.pop(ticker, None)

        selected = ranked.loc[ranked["ticker"].isin(selected_tickers)].copy().sort_values("rank")
        selected_count = len(selected)
        if selected_count > 0:
            selected["weight"] = 1.0 / selected_count
            new_weights = dict(zip(selected["ticker"], selected["weight"]))
        else:
            selected["weight"] = pd.Series(dtype=float)
            new_weights = {}

        turnover = sum(abs(new_weights.get(ticker, 0.0) - prior_weights.get(ticker, 0.0)) for ticker in set(new_weights) | set(prior_weights))
        transaction_cost = turnover * cost_bps / 10000.0
        spy_return = next_spy_price / float(spy_price) - 1 if spy_price else 0.0
        gross_return = 0.0
        buy_tickers = {
            action["ticker"]
            for action in action_rows
            if action["date"] == rebalance_date and action["action"] == "BUY"
        }

        if selected_count:
            for _, row in selected.iterrows():
                next_row = next_panel_by_ticker.loc[row["ticker"]]
                current_price = float(row["adjusted_close"])
                next_price = float(next_row["adjusted_close"])
                realized_return = next_price / current_price - 1 if current_price else 0.0
                contribution = float(row["weight"]) * realized_return
                gross_return += contribution
                attribution_rows.append(
                    {
                        "date": rebalance_date,
                        "period_end_date": next_period_date,
                        "ticker": row["ticker"],
                        "weight": float(row["weight"]),
                        "rank": int(row["rank"]),
                        "score": float(row["score"]),
                        "realized_return_while_held": realized_return,
                        "period_spy_return": spy_return,
                        "total_contribution": contribution,
                        "contribution_to_excess_return": float(row["weight"]) * (realized_return - spy_return),
                    }
                )

        net_return = gross_return - transaction_cost
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        for ticker in list(holdings):
            row = ranked_by_ticker.loc[ticker]
            action_type = "BUY" if ticker in buy_tickers else "HOLD"
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)
            holding_rows.append(
                {
                    "date": rebalance_date,
                    "period_end_date": next_period_date,
                    "strategy_name": strategy_name,
                    "ticker": ticker,
                    "action": action_type,
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "weight": float(new_weights.get(ticker, 0.0)),
                    "holding_days": int(holdings[ticker]["holding_days"]),
                }
            )

        weekly_rows.append(
            {
                "date": rebalance_date,
                "period_end_date": next_period_date,
                "strategy_name": strategy_name,
                "gross_return": gross_return,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "selected_count": selected_count,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": float(selected["weight"].sum()) if selected_count else 0.0,
            }
        )

    weekly = pd.DataFrame(weekly_rows)
    if not weekly.empty:
        weekly["model_drawdown"] = compute_drawdown(weekly["portfolio_value"])
        weekly["spy_drawdown"] = compute_drawdown(weekly["spy_value"])
    return weekly, pd.DataFrame(holding_rows), pd.DataFrame(action_rows), pd.DataFrame(attribution_rows)


def run_frozen_ml_forward(
    runtime: Config,
    candidate,
    artifact: dict[str, object],
    features_forward: pd.DataFrame,
    *,
    variant_name: str = "normal_features",
    cost_bps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = list(artifact["feature_names"])
    features_variant = apply_variant_features(features_forward, variant_name, feature_names)
    prediction_rows = features_variant.loc[features_variant["ticker"] != runtime.benchmark, ["date", "ticker", *feature_names]].copy()
    prediction_rows["predicted_score"] = artifact["estimator"].predict(prediction_rows[feature_names])
    prediction_df = prediction_rows.loc[:, ["date", "ticker", "predicted_score"]].copy()
    panels = precompute_research_panels(
        features_variant,
        runtime,
        scoring_mode="ml_prediction",
        start_date=candidate.forward_window_start,
        end_date=None,
        prediction_df=prediction_df,
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        holding_period_days=5,
    )
    return _simulate_ranked_panels(
        panels,
        top_n=int(candidate.top_n),
        cost_bps=float(candidate.total_cost_bps if cost_bps is None else cost_bps),
        enter_rank=int(candidate.enter_rank),
        hold_rank=int(candidate.hold_rank),
        max_holding_days=int(candidate.max_holding_days),
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        strategy_name=candidate.strategy_name,
    )


def run_frozen_ml_backtest_over_features(
    runtime: Config,
    candidate,
    artifact: dict[str, object],
    features: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    variant_name: str = "normal_features",
    cost_bps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = list(artifact["feature_names"])
    windowed = features.copy()
    windowed["date"] = pd.to_datetime(windowed["date"])
    windowed = windowed.loc[windowed["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        windowed = windowed.loc[windowed["date"] <= pd.Timestamp(end_date)].copy()
    features_variant = apply_variant_features(windowed, variant_name, feature_names)
    prediction_rows = features_variant.loc[features_variant["ticker"] != runtime.benchmark, ["date", "ticker", *feature_names]].copy()
    prediction_rows["predicted_score"] = artifact["estimator"].predict(prediction_rows[feature_names])
    prediction_df = prediction_rows.loc[:, ["date", "ticker", "predicted_score"]].copy()
    panels = precompute_research_panels(
        features_variant,
        runtime,
        scoring_mode="ml_prediction",
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d") if end_date is not None else None,
        prediction_df=prediction_df,
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        holding_period_days=5,
    )
    return _simulate_ranked_panels(
        panels,
        top_n=int(candidate.top_n),
        cost_bps=float(candidate.total_cost_bps if cost_bps is None else cost_bps),
        enter_rank=int(candidate.enter_rank),
        hold_rank=int(candidate.hold_rank),
        max_holding_days=int(candidate.max_holding_days),
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        strategy_name=candidate.strategy_name,
    )


def run_frozen_rule_forward(
    runtime: Config,
    features_forward: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommended = load_recommended_strategy_config(runtime.project_root)
    if recommended.strategy_name != EXPECTED_RULE_BASED:
        raise ValueError(f"recommended_strategy.yaml must remain locked to {EXPECTED_RULE_BASED}; found {recommended.strategy_name}")
    panels = precompute_recommended_low_turnover_panels(features_forward, runtime, recommended)
    return _simulate_ranked_panels(
        panels,
        top_n=int(recommended.top_n),
        cost_bps=float(recommended.total_cost_bps),
        enter_rank=int(recommended.enter_rank or recommended.top_n),
        hold_rank=int(recommended.hold_rank or recommended.top_n),
        max_holding_days=int(recommended.max_holding_days or recommended.holding_period_days),
        rebalance_frequency_days=int(recommended.rebalance_frequency_days or recommended.holding_period_days),
        strategy_name=recommended.strategy_name,
    )


def summarize_backtest_frame(weekly: pd.DataFrame) -> dict[str, float]:
    if weekly.empty:
        return {
            "total_return": float("nan"),
            "spy_return": float("nan"),
            "excess_vs_spy": float("nan"),
            "max_drawdown": float("nan"),
            "spy_max_drawdown": float("nan"),
            "average_turnover": float("nan"),
            "average_holdings": float("nan"),
            "rebalance_periods": 0,
            "periods_beating_spy": 0,
            "percent_periods_beating_spy": float("nan"),
            "estimated_trading_costs": float("nan"),
            "latest_date": pd.NaT,
            "forward_start_date": pd.NaT,
        }
    periods_beating = int((pd.to_numeric(weekly["net_return"], errors="coerce") > pd.to_numeric(weekly["spy_return"], errors="coerce")).sum())
    return {
        "total_return": float(weekly["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "spy_return": float(weekly["spy_value"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "excess_vs_spy": float(weekly["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - weekly["spy_value"].iloc[-1] / INITIAL_CAPITAL),
        "max_drawdown": float(weekly["model_drawdown"].min()),
        "spy_max_drawdown": float(weekly["spy_drawdown"].min()),
        "average_turnover": float(pd.to_numeric(weekly["turnover"], errors="coerce").mean()),
        "average_holdings": float(pd.to_numeric(weekly["selected_count"], errors="coerce").mean()),
        "rebalance_periods": int(len(weekly)),
        "periods_beating_spy": periods_beating,
        "percent_periods_beating_spy": float(periods_beating / len(weekly)),
        "estimated_trading_costs": float(pd.to_numeric(weekly["transaction_cost"], errors="coerce").sum()),
        "latest_date": pd.Timestamp(weekly["period_end_date"].max()),
        "forward_start_date": pd.Timestamp(weekly["date"].min()),
    }


def months_of_forward_data(weekly: pd.DataFrame) -> float:
    if weekly.empty:
        return 0.0
    start = pd.Timestamp(weekly["date"].min())
    end = pd.Timestamp(weekly["period_end_date"].max())
    return float((end - start).days / 30.44)
