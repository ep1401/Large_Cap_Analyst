from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_dataframe, save_dataframe


MARKET_REGIME_OUTPUT_COLUMNS = [
    "date",
    "spy_return_5d",
    "spy_return_21d",
    "spy_return_63d",
    "spy_volatility_21d",
    "spy_volatility_63d",
    "spy_above_sma_50",
    "spy_above_sma_200",
    "spy_drawdown_from_63d_high",
    "spy_drawdown_from_252d_high",
    "qqq_return_21d",
    "qqq_volatility_21d",
    "market_risk_score",
    "normalized_market_risk_score",
    "market_regime_label",
]


def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=max(20, window // 4)).mean()
    std = values.rolling(window, min_periods=max(20, window // 4)).std(ddof=0)
    out = (values - mean) / std.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)


def build_market_regime_features(
    prices_path: str | Path,
    market_sentiment_path: str | Path,
    output_path: str | Path,
    benchmark_ticker: str = "SPY",
    secondary_ticker: str = "QQQ",
) -> pd.DataFrame:
    prices = load_dataframe(prices_path, parse_dates=["date"])
    market_sentiment = load_dataframe(market_sentiment_path, parse_dates=["date"])

    spy = (
        prices.loc[prices["ticker"] == benchmark_ticker, ["date", "adjusted_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .rename(columns={"adjusted_close": "spy_adjusted_close"})
        .reset_index(drop=True)
    )
    if spy.empty:
        empty = pd.DataFrame(columns=MARKET_REGIME_OUTPUT_COLUMNS)
        save_dataframe(output_path, empty)
        return empty

    spy["spy_daily_return"] = spy["spy_adjusted_close"].pct_change()
    spy["spy_return_5d"] = spy["spy_adjusted_close"].pct_change(5)
    spy["spy_return_21d"] = spy["spy_adjusted_close"].pct_change(21)
    spy["spy_return_63d"] = spy["spy_adjusted_close"].pct_change(63)
    spy["spy_volatility_21d"] = spy["spy_daily_return"].rolling(21).std()
    spy["spy_volatility_63d"] = spy["spy_daily_return"].rolling(63).std()
    spy["spy_sma_50"] = spy["spy_adjusted_close"].rolling(50).mean()
    spy["spy_sma_200"] = spy["spy_adjusted_close"].rolling(200).mean()
    spy["spy_above_sma_50"] = (spy["spy_adjusted_close"] > spy["spy_sma_50"]).fillna(False).astype(float)
    spy["spy_above_sma_200"] = (spy["spy_adjusted_close"] > spy["spy_sma_200"]).fillna(False).astype(float)
    spy["spy_drawdown_from_63d_high"] = spy["spy_adjusted_close"] / spy["spy_adjusted_close"].rolling(63).max() - 1.0
    spy["spy_drawdown_from_252d_high"] = spy["spy_adjusted_close"] / spy["spy_adjusted_close"].rolling(252).max() - 1.0

    regime = spy.merge(market_sentiment, on="date", how="left")
    for column in [
        "market_sentiment_7d",
        "market_sentiment_30d",
        "market_sentiment_change_7d_vs_30d",
        "market_negative_news_ratio_7d",
        "percent_tickers_positive_sentiment_7d",
        "percent_tickers_negative_sentiment_7d",
        "market_news_breadth_7d",
        "sentiment_dispersion_7d",
    ]:
        if column not in regime.columns:
            regime[column] = 0.0
        regime[column] = pd.to_numeric(regime[column], errors="coerce").fillna(0.0)

    qqq = (
        prices.loc[prices["ticker"] == secondary_ticker, ["date", "adjusted_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .rename(columns={"adjusted_close": "qqq_adjusted_close"})
        .reset_index(drop=True)
    )
    if not qqq.empty:
        qqq["qqq_daily_return"] = qqq["qqq_adjusted_close"].pct_change()
        qqq["qqq_return_21d"] = qqq["qqq_adjusted_close"].pct_change(21)
        qqq["qqq_volatility_21d"] = qqq["qqq_daily_return"].rolling(21).std()
        regime = regime.merge(qqq[["date", "qqq_return_21d", "qqq_volatility_21d"]], on="date", how="left")
    else:
        regime["qqq_return_21d"] = np.nan
        regime["qqq_volatility_21d"] = np.nan

    raw_risk_score = (
        0.18 * _rolling_zscore(regime["spy_return_21d"])
        + 0.12 * _rolling_zscore(regime["spy_return_63d"])
        + 0.08 * _rolling_zscore(regime["market_sentiment_7d"])
        + 0.06 * _rolling_zscore(regime["market_sentiment_change_7d_vs_30d"])
        + 0.08 * _rolling_zscore(regime["market_news_breadth_7d"])
        + 0.08 * _rolling_zscore(regime["percent_tickers_positive_sentiment_7d"] - regime["percent_tickers_negative_sentiment_7d"])
        - 0.12 * _rolling_zscore(regime["spy_volatility_21d"])
        - 0.08 * _rolling_zscore(regime["spy_volatility_63d"])
        - 0.10 * _rolling_zscore(regime["spy_drawdown_from_63d_high"].abs())
        - 0.08 * _rolling_zscore(regime["spy_drawdown_from_252d_high"].abs())
        - 0.06 * _rolling_zscore(regime["market_negative_news_ratio_7d"])
        - 0.04 * _rolling_zscore(regime["sentiment_dispersion_7d"])
        + 0.05 * (regime["spy_above_sma_50"] * 2 - 1)
        + 0.05 * (regime["spy_above_sma_200"] * 2 - 1)
    )
    if regime["qqq_return_21d"].notna().any():
        raw_risk_score = raw_risk_score + 0.08 * _rolling_zscore(regime["qqq_return_21d"]) - 0.06 * _rolling_zscore(
            regime["qqq_volatility_21d"]
        )

    regime["market_risk_score"] = np.tanh(raw_risk_score.fillna(0.0))
    regime["normalized_market_risk_score"] = ((regime["market_risk_score"] + 1.0) / 2.0).clip(0.0, 1.0)
    regime["market_regime_label"] = np.where(
        regime["market_risk_score"] >= 0.20,
        "risk_on",
        np.where(regime["market_risk_score"] <= -0.20, "risk_off", "neutral"),
    )

    output = regime[MARKET_REGIME_OUTPUT_COLUMNS].copy()
    save_dataframe(output_path, output)
    return output
