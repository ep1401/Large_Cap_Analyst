from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest import select_rebalance_dates
from src.config import Config
from src.recommended_strategy import load_promoted_tuned_weights
from src.scoring import score_rebalance_date
from src.utils import save_dataframe


RESEARCH_CAVEAT_LINES = [
    "This is a research candidate workflow.",
    "2026 forward data is not used for tuning or model selection.",
    "Back-tested performance is hypothetical.",
    "Snapshot analyst target fields are excluded.",
    "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
    "News sentiment depends on Alpha Vantage coverage and classification.",
    "ML models may overfit and require future forward validation.",
    "This is research/paper trading only, not financial advice.",
]

ML_ALLOWED_FEATURES = [
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "historical_rating_score_change_30d",
    "net_upgrade_score_30d",
    "downgrade_count_30d",
    "recent_downgrade_flag_30d",
    "relevance_weighted_sentiment_7d",
    "relevance_weighted_sentiment_30d",
    "sentiment_change_7d_vs_30d",
    "negative_news_ratio_7d",
    "relative_strength_21d",
    "relative_strength_63d",
    "volatility_21d",
    "beta_to_spy_63d",
    "distance_to_63d_high",
    "breakout_63d",
    "market_sentiment_7d",
    "market_sentiment_30d",
    "market_negative_news_ratio_7d",
    "percent_tickers_positive_sentiment_7d",
    "percent_tickers_negative_sentiment_7d",
    "market_risk_score",
    "spy_return_21d",
    "spy_volatility_21d",
    "spy_drawdown_from_63d_high",
    "spy_above_sma_50",
    "spy_above_sma_200",
]
ML_TARGET_COLUMN = "future_5d_excess_return"
RESEARCH_STRATEGY_FEATURES = {
    "final_quant_5d_weight_tuned_market_regime_no_snapshot": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "relative_strength_21d",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "volatility_21d",
        "breakout_63d",
        "recent_downgrade_flag_30d",
        "strong_negative_news_flag",
        "market_risk_score",
        "market_regime_label",
        "market_sentiment_7d",
        "spy_volatility_21d",
        "spy_drawdown_from_63d_high",
        "spy_above_sma_50",
        "spy_above_sma_200",
    },
    "final_quant_5d_market_aware_score_no_snapshot": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "relative_strength_21d",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "volatility_21d",
        "breakout_63d",
        "recent_downgrade_flag_30d",
        "strong_negative_news_flag",
        "negative_news_ratio_7d",
        "market_sentiment_7d",
        "spy_return_21d",
        "spy_volatility_21d",
    },
    "ml_ranker_5d_no_snapshot": set(ML_ALLOWED_FEATURES),
    "ml_ranker_5d_market_exposure_no_snapshot": set(ML_ALLOWED_FEATURES)
    | {"market_risk_score", "market_regime_label", "normalized_market_risk_score"},
}


@dataclass(slots=True)
class MLResearchCandidateConfig:
    strategy_name: str
    model_type: str
    model_path: str
    training_window: str
    validation_window: str
    forward_window_start: str
    execution_mode: str
    enter_rank: int
    hold_rank: int
    top_n: int
    max_holding_days: int
    rebalance_frequency_days: int
    total_cost_bps: float
    position_sizing: str
    long_short: bool
    use_regime_filter: bool
    snapshot_fields_allowed: bool
    status: str


def _parse_scalar(value: str) -> str | int | float | bool | None:
    text = value.strip()
    if text in {"null", "None", "none", "~"}:
        return None
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def ml_research_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "ml_research_candidate.yaml"


def load_ml_research_candidate_config(project_root: Path | None = None) -> MLResearchCandidateConfig:
    root = project_root or Path(__file__).resolve().parents[1]
    path = ml_research_config_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing ML research candidate config: {path}")
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return MLResearchCandidateConfig(**data)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return ((values - values.mean()) / std).clip(-3.0, 3.0).fillna(0.0)


def qualify_research_universe(day: pd.DataFrame, runtime: Config) -> pd.DataFrame:
    qualified = day.copy()
    mask = pd.Series(True, index=qualified.index)
    mask &= qualified["adjusted_close"].notna()
    mask &= pd.to_numeric(qualified["avg_dollar_volume_21d"], errors="coerce").fillna(0.0) >= runtime.min_avg_dollar_volume
    mask &= qualified["historical_rating_count_data_available"].fillna(False).astype(bool)
    mask &= pd.to_numeric(qualified["historical_total_ratings"], errors="coerce").fillna(0.0) >= 5
    mask &= qualified["historical_grade_data_available"].fillna(False).astype(bool)
    return qualified.loc[mask].copy()


def score_base_tuned(day: pd.DataFrame) -> pd.DataFrame:
    return score_rebalance_date(
        day.copy(),
        strategy_name="final_quant_5d_weight_tuned_no_snapshot",
        use_analyst_filters=False,
        resistance_window=30,
    )


def score_market_aware(day: pd.DataFrame) -> pd.DataFrame:
    scored = score_base_tuned(day)
    market_sentiment_positive = pd.to_numeric(scored["market_sentiment_7d"], errors="coerce").fillna(0.0).clip(lower=0.0)
    market_sentiment_negative = (-pd.to_numeric(scored["market_sentiment_7d"], errors="coerce").fillna(0.0)).clip(lower=0.0)
    spy_return_21d = pd.to_numeric(scored["spy_return_21d"], errors="coerce").fillna(0.0)
    spy_volatility_21d = pd.to_numeric(scored["spy_volatility_21d"], errors="coerce").fillna(0.0)

    scored["score"] = (
        pd.to_numeric(scored["score"], errors="coerce").fillna(0.0)
        + 0.10 * cross_sectional_zscore(pd.to_numeric(scored["relevance_weighted_sentiment_7d"], errors="coerce").fillna(0.0) * market_sentiment_positive)
        - 0.10 * cross_sectional_zscore(pd.to_numeric(scored["negative_news_ratio_7d"], errors="coerce").fillna(0.0) * market_sentiment_negative)
        + 0.05 * cross_sectional_zscore(pd.to_numeric(scored["relative_strength_21d"], errors="coerce").fillna(0.0) * spy_return_21d)
        - 0.05 * cross_sectional_zscore(pd.to_numeric(scored["volatility_21d"], errors="coerce").fillna(0.0) * spy_volatility_21d)
    )
    return scored


def market_exposure_value(panel: pd.DataFrame, exposure_mode: str) -> float:
    if panel.empty:
        return 0.0
    if exposure_mode == "full":
        return 1.0
    if exposure_mode == "discrete":
        label = str(panel["market_regime_label"].iloc[0]) if "market_regime_label" in panel.columns else "neutral"
        return {"risk_on": 1.0, "neutral": 0.75, "risk_off": 0.50}.get(label, 0.75)
    if exposure_mode == "continuous":
        normalized = pd.to_numeric(panel.get("normalized_market_risk_score"), errors="coerce").fillna(0.5)
        if isinstance(normalized, pd.Series):
            normalized_value = float(normalized.iloc[0])
        else:
            normalized_value = float(normalized)
        return float(np.clip(0.5 + 0.5 * normalized_value, 0.4, 1.0))
    raise ValueError(f"Unsupported exposure_mode: {exposure_mode}")


def precompute_research_panels(
    features: pd.DataFrame,
    runtime: Config,
    *,
    scoring_mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
    prediction_df: pd.DataFrame | None = None,
    rebalance_frequency_days: int = 15,
    holding_period_days: int = 5,
) -> list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]]:
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start_date is not None:
        df = df.loc[df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        df = df.loc[df["date"] <= pd.Timestamp(end_date)].copy()
    rebalance_dates = select_rebalance_dates(
        df,
        holding_period_days=holding_period_days,
        benchmark=runtime.benchmark,
        rebalance_frequency_days=rebalance_frequency_days,
    )
    available_dates = sorted(pd.to_datetime(df["date"]).drop_duplicates())
    if available_dates and rebalance_dates and pd.Timestamp(rebalance_dates[-1]) < pd.Timestamp(available_dates[-1]):
        rebalance_dates = [*rebalance_dates, pd.Timestamp(available_dates[-1])]

    if prediction_df is not None and not prediction_df.empty:
        predictions = prediction_df.copy()
        predictions["date"] = pd.to_datetime(predictions["date"])
    else:
        predictions = pd.DataFrame(columns=["date", "ticker", "predicted_score"])

    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]] = []
    for rebalance_date in rebalance_dates:
        day_all = df.loc[df["date"] == rebalance_date].copy()
        benchmark_slice = day_all.loc[day_all["ticker"] == runtime.benchmark, "adjusted_close"]
        if benchmark_slice.empty or pd.isna(benchmark_slice.iloc[0]):
            continue
        day = qualify_research_universe(day_all.loc[day_all["ticker"] != runtime.benchmark].copy(), runtime)
        if day.empty:
            continue
        if scoring_mode == "base_tuned":
            scored = score_base_tuned(day)
        elif scoring_mode == "market_aware":
            scored = score_market_aware(day)
        elif scoring_mode == "ml_prediction":
            scored = day.merge(
                predictions.loc[predictions["date"] == rebalance_date, ["ticker", "predicted_score"]],
                on="ticker",
                how="left",
            )
            scored["score"] = pd.to_numeric(scored["predicted_score"], errors="coerce").fillna(-np.inf)
        else:
            raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")
        scored["strong_negative_news_flag"] = scored["strong_negative_news_flag"].fillna(False).astype(bool)
        scored["recent_downgrade_flag_30d"] = scored["recent_downgrade_flag_30d"].fillna(False).astype(bool)
        price_lookup = day_all.loc[day_all["ticker"] != runtime.benchmark, ["ticker", "adjusted_close"]].copy()
        panels.append((pd.Timestamp(rebalance_date), scored.copy(), float(benchmark_slice.iloc[0]), price_lookup))
    return panels


def run_low_turnover_research_backtest(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    *,
    top_n: int = 10,
    cost_bps: float = 20.0,
    enter_rank: int = 10,
    hold_rank: int = 20,
    max_holding_days: int = 21,
    rebalance_frequency_days: int = 15,
    strategy_name: str,
    exposure_mode: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    portfolio_value = 10000.0
    spy_value = 10000.0

    if len(panels) < 2:
        return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows)

    for idx in range(len(panels) - 1):
        rebalance_date, panel, spy_price, _ = panels[idx]
        prior_weights = {ticker: float(meta.get("weight", 0.0)) for ticker, meta in holdings.items()}
        ranked = panel.sort_values("score", ascending=False).reset_index(drop=True).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_by_ticker = ranked.set_index("ticker", drop=False)
        next_panel_by_ticker = panels[idx + 1][3].set_index("ticker", drop=False)
        next_spy_price = float(panels[idx + 1][2])

        forced_sells: list[str] = []
        discretionary_sells: list[tuple[str, int]] = []
        for ticker, meta in list(holdings.items()):
            if ticker not in ranked_by_ticker.index:
                forced_sells.append(ticker)
                continue
            row = ranked_by_ticker.loc[ticker]
            rank = int(row["rank"])
            if bool(row["strong_negative_news_flag"]) or bool(row["recent_downgrade_flag_30d"]) or int(meta["holding_days"]) >= max_holding_days:
                forced_sells.append(ticker)
            elif rank > hold_rank:
                discretionary_sells.append((ticker, rank))

        for ticker in forced_sells:
            holdings.pop(ticker, None)
        for ticker, _ in sorted(discretionary_sells, key=lambda item: item[1], reverse=True):
            holdings.pop(ticker, None)

        current_tickers = list(holdings.keys())
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_tickers
        ]
        open_slots = max(0, top_n - len(current_tickers))
        for ticker in desired_buys[:open_slots]:
            holdings[ticker] = {"holding_days": 0, "entry_date": rebalance_date}

        selected_tickers = [ticker for ticker in ranked["ticker"].tolist() if ticker in holdings][:top_n]
        for ticker in list(holdings):
            if ticker not in selected_tickers:
                holdings.pop(ticker, None)

        selected = ranked.loc[ranked["ticker"].isin(selected_tickers)].copy().sort_values("rank")
        target_exposure = market_exposure_value(selected if not selected.empty else ranked.head(1), exposure_mode)
        selected_count = len(selected)
        if selected_count > 0:
            weight = target_exposure / selected_count
            selected["weight"] = weight
            new_weights = dict(zip(selected["ticker"], selected["weight"]))
        else:
            selected["weight"] = pd.Series(dtype=float)
            new_weights = {}

        turnover = sum(abs(new_weights.get(ticker, 0.0) - prior_weights.get(ticker, 0.0)) for ticker in set(new_weights) | set(prior_weights))
        transaction_cost = turnover * cost_bps / 10000.0
        gross_return = 0.0
        if selected_count:
            for _, row in selected.iterrows():
                if row["ticker"] in next_panel_by_ticker.index:
                    next_row = next_panel_by_ticker.loc[row["ticker"]]
                    current_price = float(row["adjusted_close"])
                    next_price = float(next_row["adjusted_close"])
                    realized_return = next_price / current_price - 1 if current_price else 0.0
                else:
                    realized_return = 0.0
                gross_return += float(row["weight"]) * realized_return
        net_return = gross_return - transaction_cost
        spy_return = next_spy_price / float(spy_price) - 1 if spy_price else 0.0
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        for ticker in list(holdings):
            row = ranked_by_ticker.loc[ticker]
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)
            holdings[ticker]["weight"] = float(new_weights.get(ticker, 0.0))
            holding_rows.append(
                {
                    "date": rebalance_date,
                    "ticker": ticker,
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "weight": float(new_weights.get(ticker, 0.0)),
                    "target_exposure": target_exposure,
                    "holding_days": int(holdings[ticker]["holding_days"]),
                }
            )

        weekly_rows.append(
            {
                "date": rebalance_date,
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
                "target_exposure": target_exposure,
            }
        )

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows)


def compute_rank_spread_for_scores(scored_frames: list[pd.DataFrame], target_column: str = ML_TARGET_COLUMN) -> dict[str, float]:
    per_date_rows: list[dict[str, float]] = []
    for scored in scored_frames:
        if scored.empty or target_column not in scored.columns:
            continue
        ranked = scored.sort_values("score", ascending=False).copy()
        if len(ranked) < 10:
            continue
        top_n = max(1, int(np.ceil(len(ranked) * 0.10)))
        bottom_n = max(1, int(np.ceil(len(ranked) * 0.10)))
        top_excess = float(pd.to_numeric(ranked.head(top_n)[target_column], errors="coerce").mean())
        bottom_excess = float(pd.to_numeric(ranked.tail(bottom_n)[target_column], errors="coerce").mean())
        rank_corr = float(
            pd.to_numeric(ranked["score"], errors="coerce").corr(pd.to_numeric(ranked[target_column], errors="coerce"), method="spearman")
        )
        per_date_rows.append(
            {
                "top_decile_avg_excess": top_excess,
                "bottom_decile_avg_excess": bottom_excess,
                "top_minus_bottom_spread": top_excess - bottom_excess,
                "rank_correlation": rank_corr,
            }
        )
    spread_df = pd.DataFrame(per_date_rows)
    if spread_df.empty:
        return {
            "rank_correlation": float("nan"),
            "top_decile_avg_excess": float("nan"),
            "bottom_decile_avg_excess": float("nan"),
            "top_minus_bottom_spread": float("nan"),
        }
    return {column: float(pd.to_numeric(spread_df[column], errors="coerce").mean()) for column in spread_df.columns}


def build_ml_estimators() -> dict[str, Pipeline]:
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return {
        "ridge_regression": Pipeline(
            [
                ("preprocess", ColumnTransformer([("num", numeric_preprocessor, ML_ALLOWED_FEATURES)], remainder="drop")),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_regression": Pipeline(
            [
                ("preprocess", ColumnTransformer([("num", SimpleImputer(strategy="median"), ML_ALLOWED_FEATURES)], remainder="drop")),
                ("model", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1)),
            ]
        ),
        "hist_gradient_boosting_regression": Pipeline(
            [
                ("preprocess", ColumnTransformer([("num", SimpleImputer(strategy="median"), ML_ALLOWED_FEATURES)], remainder="drop")),
                ("model", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=42)),
            ]
        ),
        "logistic_outperform_classifier": Pipeline(
            [
                ("preprocess", ColumnTransformer([("num", numeric_preprocessor, ML_ALLOWED_FEATURES)], remainder="drop")),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
    }


def fit_and_score_ml_model(
    model_name: str,
    estimator: Pipeline,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Pipeline]:
    x_train = train_df[ML_ALLOWED_FEATURES].copy()
    y_train = pd.to_numeric(train_df[ML_TARGET_COLUMN], errors="coerce").fillna(0.0)
    x_val = validation_df[ML_ALLOWED_FEATURES].copy()

    if model_name == "logistic_outperform_classifier":
        estimator.fit(x_train, (y_train > 0).astype(int))
        predicted_score = estimator.predict_proba(x_val)[:, 1]
    else:
        estimator.fit(x_train, y_train)
        predicted_score = estimator.predict(x_val)

    predictions = validation_df[["date", "ticker", ML_TARGET_COLUMN, "future_5d_return", "future_5d_spy_return"]].copy()
    predictions["predicted_score"] = predicted_score
    predictions["model_name"] = model_name
    return predictions, estimator


def save_ml_artifact(
    artifact_path: str | Path,
    estimator: Pipeline,
    model_name: str,
    predictions_2025: pd.DataFrame,
) -> None:
    artifact = {
        "model_name": model_name,
        "target_column": ML_TARGET_COLUMN,
        "feature_names": ML_ALLOWED_FEATURES,
        "train_start_date": "2023-01-01",
        "train_end_date": "2024-12-31",
        "validation_start_date": "2025-01-01",
        "validation_end_date": "2025-12-31",
        "estimator": estimator,
        "predictions_2025": predictions_2025,
    }
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)


def load_ml_artifact(path: str | Path) -> dict[str, object]:
    return joblib.load(path)


def build_scored_frames_from_predictions(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[pd.DataFrame]:
    merged = features.merge(predictions[["date", "ticker", "predicted_score"]], on=["date", "ticker"], how="inner")
    frames: list[pd.DataFrame] = []
    for _, day in merged.groupby("date"):
        if day.empty:
            continue
        scored = day.copy()
        scored["score"] = pd.to_numeric(scored["predicted_score"], errors="coerce")
        frames.append(scored)
    return frames
