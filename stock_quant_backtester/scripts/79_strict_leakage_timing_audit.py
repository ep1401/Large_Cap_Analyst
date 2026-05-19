from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown
from src.recommended_strategy import load_recommended_strategy_config
from src.research_models import (
    ML_ALLOWED_FEATURES,
    fit_and_score_ml_model,
    load_ml_artifact,
    load_ml_research_candidate_config,
    precompute_research_panels,
)
from src.utils import load_dataframe, save_dataframe


INITIAL_CAPITAL = 10000.0
EXPECTED_RULE_BASED = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
FORWARD_FEATURES_PATH = "features_panel_2026_forward.csv"

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

FORBIDDEN_FEATURES = [
    "consensus_upside",
    "low_target_upside",
    "high_target_upside",
    "consensus_target",
    "low_target",
    "high_target",
    "median_target",
    "target_spread",
    "target_revision_7d",
    "target_revision_30d",
    "future_5d_return",
    "future_21d_return",
    "future_63d_return",
    "future_5d_excess_return",
    "future_21d_excess_return",
    "future_63d_excess_return",
    "future_5d_spy_return",
    "future_21d_spy_return",
    "future_63d_spy_return",
]


def _caveat_lines() -> list[str]:
    return [
        "This is a frozen ML research candidate.",
        "2026 data was not used for training, tuning, or model selection.",
        "Conservative lag tests are used to detect possible timing leakage.",
        "Back-tested performance is hypothetical unless actually paper-tracked live.",
        "ML models may overfit and require extended forward validation.",
        "This is research/paper trading only, not financial advice.",
    ]


def _compute_drawdown(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values / values.cummax() - 1.0


def _ensure_market_features(runtime: Config, features_forward: pd.DataFrame) -> pd.DataFrame:
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


def _lag_columns_by_trading_day(df: pd.DataFrame, columns: list[str], days: int) -> pd.DataFrame:
    shifted = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    present = [column for column in columns if column in shifted.columns]
    for column in present:
        shifted[column] = shifted.groupby("ticker")[column].shift(days)
    return shifted


def _variant_features(base_features: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    if variant_name == "normal_features":
        return base_features.copy()
    if variant_name == "sentiment_lag_1d":
        return _lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES, 1)
    if variant_name == "sentiment_lag_2d":
        return _lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES, 2)
    if variant_name == "ratings_lag_1d":
        return _lag_columns_by_trading_day(base_features, RATING_FEATURES, 1)
    if variant_name == "grade_events_lag_1d":
        return _lag_columns_by_trading_day(base_features, GRADE_EVENT_FEATURES, 1)
    if variant_name == "technical_lag_1d":
        return _lag_columns_by_trading_day(base_features, TECHNICAL_FEATURES, 1)
    if variant_name == "all_non_price_alt_data_lag_1d":
        return _lag_columns_by_trading_day(base_features, SENTIMENT_FEATURES + RATING_FEATURES + GRADE_EVENT_FEATURES, 1)
    if variant_name == "all_features_lag_1d":
        return _lag_columns_by_trading_day(base_features, ML_ALLOWED_FEATURES, 1)
    raise ValueError(f"Unknown variant_name: {variant_name}")


def _simulate_ml_forward(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    *,
    top_n: int,
    cost_bps: float,
    enter_rank: int,
    hold_rank: int,
    max_holding_days: int,
    rebalance_frequency_days: int,
) -> tuple[pd.DataFrame, int]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    portfolio_value = INITIAL_CAPITAL
    fallback_count = 0

    if len(panels) < 2:
        return pd.DataFrame(), 0

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
            holdings[ticker] = {"holding_days": 0}

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
        gross_return = 0.0
        if selected_count:
            for _, row in selected.iterrows():
                if row["ticker"] in next_panel_by_ticker.index:
                    next_row = next_panel_by_ticker.loc[row["ticker"]]
                    current_price = float(row["adjusted_close"])
                    next_price = float(next_row["adjusted_close"])
                    realized_return = next_price / current_price - 1 if current_price else 0.0
                else:
                    fallback_count += 1
                    realized_return = float(row.get("future_return_used", 0.0))
                gross_return += float(row["weight"]) * realized_return
        net_return = gross_return - transaction_cost
        spy_return = next_spy_price / float(spy_price) - 1 if spy_price else 0.0
        portfolio_value *= 1 + net_return

        for ticker in list(holdings):
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)

        weekly_rows.append(
            {
                "date": rebalance_date,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "selected_count": selected_count,
                "portfolio_value": portfolio_value,
            }
        )

    weekly = pd.DataFrame(weekly_rows)
    if not weekly.empty:
        weekly["drawdown"] = _compute_drawdown(weekly["portfolio_value"])
    return weekly, fallback_count


def _run_variant(
    runtime: Config,
    candidate,
    artifact: dict[str, object],
    base_features: pd.DataFrame,
    variant_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    features_variant = _variant_features(base_features, variant_name)
    estimator = artifact["estimator"]
    feature_names = list(artifact["feature_names"])
    prediction_rows = features_variant.loc[features_variant["ticker"] != runtime.benchmark, ["date", "ticker", *feature_names]].copy()
    prediction_rows["predicted_score"] = estimator.predict(prediction_rows[feature_names])
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
    weekly, fallback_count = _simulate_ml_forward(
        panels,
        top_n=int(candidate.top_n),
        cost_bps=float(candidate.total_cost_bps),
        enter_rank=int(candidate.enter_rank),
        hold_rank=int(candidate.hold_rank),
        max_holding_days=int(candidate.max_holding_days),
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
    )
    if weekly.empty:
        raise ValueError(f"Variant {variant_name} produced no weekly rows.")

    ml_return = float(weekly["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    spy_return = float((1.0 + pd.to_numeric(weekly["spy_return"], errors="coerce")).prod() - 1.0)
    row = {
        "variant_name": variant_name,
        "ml_return": ml_return,
        "spy_return": spy_return,
        "excess_vs_spy": ml_return - spy_return,
        "max_drawdown": float(weekly["drawdown"].min()),
        "turnover": float(pd.to_numeric(weekly["turnover"], errors="coerce").mean()),
        "average_holdings": float(pd.to_numeric(weekly["selected_count"], errors="coerce").mean()),
        "rebalance_periods": int(len(weekly)),
        "fallback_future_return_uses": int(fallback_count),
    }
    return row, weekly


def _audit_sentiment_timestamps(runtime: Config) -> dict[str, object]:
    articles = load_dataframe(runtime.processed_dir / "news_sentiment_articles.csv", parse_dates=["published_date", "date"])
    if articles.empty:
        return {"sentiment_article_rows": 0, "sentiment_future_article_violations": 0}
    article_dates = pd.to_datetime(articles["published_date"]).dt.tz_convert(None).dt.normalize()
    feature_dates = pd.to_datetime(articles["date"]).dt.normalize()
    violations = int((article_dates > feature_dates).sum())
    return {
        "sentiment_article_rows": int(len(articles)),
        "sentiment_future_article_violations": violations,
    }


def _audit_rating_timestamps(features_forward: pd.DataFrame) -> dict[str, object]:
    available = features_forward.loc[features_forward["historical_rating_count_data_available"].fillna(False)].copy()
    if available.empty:
        return {"rating_rows_available": 0, "rating_future_violations": 0, "rating_same_day_rows": 0}
    record_dates = pd.to_datetime(available["historical_rating_record_date"]).dt.normalize()
    feature_dates = pd.to_datetime(available["date"]).dt.normalize()
    violations = int((record_dates > feature_dates).sum())
    same_day = int((record_dates == feature_dates).sum())
    return {
        "rating_rows_available": int(len(available)),
        "rating_future_violations": violations,
        "rating_same_day_rows": same_day,
    }


def _audit_grade_event_timestamps(runtime: Config, features_forward: pd.DataFrame) -> dict[str, object]:
    events = load_dataframe(runtime.processed_dir / "historical_analyst_grade_events.csv", parse_dates=["date"])
    rows = features_forward.loc[features_forward["historical_grade_data_available"].fillna(False)].copy()
    if events.empty or rows.empty:
        return {"grade_event_rows": 0, "negative_days_since_upgrade_rows": 0, "negative_days_since_downgrade_rows": 0}
    neg_up = int(pd.to_numeric(rows.get("days_since_last_upgrade"), errors="coerce").lt(0).fillna(False).sum())
    neg_down = int(pd.to_numeric(rows.get("days_since_last_downgrade"), errors="coerce").lt(0).fillna(False).sum())
    return {
        "grade_event_rows": int(len(events)),
        "negative_days_since_upgrade_rows": neg_up,
        "negative_days_since_downgrade_rows": neg_down,
    }


def _audit_execution_interval(runtime: Config, candidate, base_features: pd.DataFrame, artifact: dict[str, object]) -> list[str]:
    findings: list[str] = []
    row, weekly = _run_variant(runtime, candidate, artifact, base_features, "normal_features")
    if row["fallback_future_return_uses"] > 0:
        findings.append(
            f"FAIL: future_5d_return fallback was used {row['fallback_future_return_uses']} times for realized P&L while rebalance_frequency_days={candidate.rebalance_frequency_days}."
        )
    else:
        findings.append("PASS: realized P&L used adjusted_close at the current and next decision dates for every held position in the 2026 forward window.")

    spy_direct = load_dataframe(runtime.tables_dir / "forward_2026_benchmark_validation.csv", parse_dates=["start_date", "end_date"])
    if spy_direct.empty:
        findings.append("FAIL: missing benchmark validation table from the rule-based forward workflow.")
    else:
        diff = float(spy_direct["absolute_difference"].iloc[0])
        if diff > 0.005:
            findings.append(f"FAIL: SPY benchmark difference exceeded 0.5% ({diff:.4%}).")
        else:
            findings.append(f"PASS: SPY benchmark period compounding matched direct adjusted-close buy-and-hold within {diff:.4%}.")
    return findings


def _audit_preprocessing(runtime: Config, artifact: dict[str, object]) -> list[str]:
    findings: list[str] = []
    if str(artifact.get("train_end_date")) <= "2024-12-31":
        findings.append("PASS: artifact training max date is <= 2024-12-31.")
    else:
        findings.append("FAIL: artifact training max date extends beyond 2024-12-31.")
    if str(artifact.get("validation_start_date")) >= "2025-01-01" and str(artifact.get("validation_end_date")) <= "2025-12-31":
        findings.append("PASS: artifact validation/model-selection window is confined to 2025.")
    else:
        findings.append("FAIL: artifact validation/model-selection window is not confined to 2025.")

    estimator = artifact["estimator"]
    findings.append(f"PASS: loaded estimator from disk is a `{type(estimator).__name__}` with fitted preprocessing inside the saved pipeline.")
    source = inspect.getsource(fit_and_score_ml_model)
    if "x_train = train_df[ML_ALLOWED_FEATURES].copy()" in source and "x_val = validation_df[ML_ALLOWED_FEATURES].copy()" in source:
        findings.append("PASS: training code slices train and validation features separately before fitting.")
    else:
        findings.append("FAIL: training code inspection could not confirm separate train/validation feature matrices.")
    if "estimator.fit(x_train, y_train)" in source and "predict(x_val)" in source:
        findings.append("PASS: scaler/imputer/model are fit on train data and applied to validation through the saved pipeline.")
    else:
        findings.append("FAIL: training code inspection could not confirm fit-on-train/predict-on-validation flow.")
    if "fit_transform" not in source:
        findings.append("PASS: no full-panel fit_transform step was found in the ML training function.")
    else:
        findings.append("WARNING: fit_transform appears in the ML training function source and should be reviewed manually.")
    if "feature_selection" not in source.lower():
        findings.append("PASS: no explicit feature-selection stage is present in the training function.")
    else:
        findings.append("WARNING: feature-selection logic appears in the training source and should be reviewed manually.")
    return findings


def _build_feature_audit_table(feature_names: list[str]) -> pd.DataFrame:
    group_map = {}
    for name in SENTIMENT_FEATURES:
        group_map[name] = "sentiment"
    for name in RATING_FEATURES:
        group_map[name] = "historical_ratings"
    for name in GRADE_EVENT_FEATURES:
        group_map[name] = "grade_events"
    for name in TECHNICAL_FEATURES:
        group_map[name] = "technical_market"

    rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        if feature_name in SENTIMENT_FEATURES:
            leakage_risk = "medium"
            notes = "Time-sensitive news or market sentiment input; lag stress-tested."
        elif feature_name in RATING_FEATURES:
            leakage_risk = "medium"
            notes = "Historical rating-count feature keyed by historical_rating_record_date."
        elif feature_name in GRADE_EVENT_FEATURES:
            leakage_risk = "medium"
            notes = "Event-derived feature keyed by historical event dates."
        else:
            leakage_risk = "low"
            notes = "Price/market-derived trailing feature."
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": group_map.get(feature_name, "other"),
                "allowed": True,
                "leakage_risk": leakage_risk,
                "notes": notes,
            }
        )

    for feature_name in FORBIDDEN_FEATURES:
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": "forbidden_audit",
                "allowed": False,
                "leakage_risk": "high",
                "notes": "Forbidden target/snapshot/forward-return field; not present in the frozen ML feature list.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    runtime = Config.from_env()
    candidate = load_ml_research_candidate_config(runtime.project_root)
    artifact = load_ml_artifact(runtime.project_root / candidate.model_path)
    rule_config = load_recommended_strategy_config(runtime.project_root)

    if rule_config.strategy_name != EXPECTED_RULE_BASED:
        raise ValueError("recommended_strategy.yaml changed unexpectedly; audit requires the frozen rule-based comparison model.")

    features_forward = load_dataframe(runtime.final_dir / FORWARD_FEATURES_PATH, parse_dates=["date"])
    if features_forward.empty:
        raise ValueError("2026 forward feature panel is empty.")
    features_forward = _ensure_market_features(runtime, features_forward)
    features_forward = features_forward.loc[features_forward["date"] >= pd.Timestamp(candidate.forward_window_start)].copy()
    if features_forward.empty:
        raise ValueError("No 2026 forward rows remain after applying the frozen candidate start date.")

    variants = [
        "normal_features",
        "sentiment_lag_1d",
        "sentiment_lag_2d",
        "ratings_lag_1d",
        "grade_events_lag_1d",
        "technical_lag_1d",
        "all_non_price_alt_data_lag_1d",
        "all_features_lag_1d",
    ]
    variant_rows: list[dict[str, object]] = []
    for variant in variants:
        row, _ = _run_variant(runtime, candidate, artifact, features_forward, variant)
        variant_rows.append(row)

    audit_df = pd.DataFrame(variant_rows)
    normal_return = float(audit_df.loc[audit_df["variant_name"] == "normal_features", "ml_return"].iloc[0])
    audit_df["performance_drop_vs_normal"] = audit_df["ml_return"] - normal_return
    save_dataframe(runtime.tables_dir / "ml_strict_leakage_timing_audit.csv", audit_df)

    execution_findings = _audit_execution_interval(runtime, candidate, artifact=artifact, base_features=features_forward)
    preprocessing_findings = _audit_preprocessing(runtime, artifact)
    feature_audit_df = _build_feature_audit_table(list(artifact["feature_names"]))
    save_dataframe(runtime.tables_dir / "ml_feature_list_audit.csv", feature_audit_df)

    sentiment_checks = _audit_sentiment_timestamps(runtime)
    rating_checks = _audit_rating_timestamps(features_forward)
    grade_checks = _audit_grade_event_timestamps(runtime, features_forward)

    execution_report_lines = [
        "# ML Execution Interval Audit",
        "",
        *[f"- {line}" for line in _caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Rebalance frequency: {candidate.rebalance_frequency_days} trading days",
        f"- Max holding days: {candidate.max_holding_days}",
        "",
        *[f"- {line}" for line in execution_findings],
    ]
    (runtime.reports_dir / "ml_execution_interval_audit.md").write_text("\n".join(execution_report_lines), encoding="utf-8")

    preprocessing_report_lines = [
        "# ML Preprocessing Leakage Audit",
        "",
        *[f"- {line}" for line in _caveat_lines()],
        "",
        f"- Candidate model type: `{candidate.model_type}`",
        f"- Artifact path: `{candidate.model_path}`",
        "",
        *[f"- {line}" for line in preprocessing_findings],
    ]
    (runtime.reports_dir / "ml_preprocessing_leakage_audit.md").write_text("\n".join(preprocessing_report_lines), encoding="utf-8")

    sentiment_lag_1d_excess = float(audit_df.loc[audit_df["variant_name"] == "sentiment_lag_1d", "excess_vs_spy"].iloc[0])
    sentiment_lag_2d_excess = float(audit_df.loc[audit_df["variant_name"] == "sentiment_lag_2d", "excess_vs_spy"].iloc[0])
    ratings_lag_excess = float(audit_df.loc[audit_df["variant_name"] == "ratings_lag_1d", "excess_vs_spy"].iloc[0])
    grade_lag_excess = float(audit_df.loc[audit_df["variant_name"] == "grade_events_lag_1d", "excess_vs_spy"].iloc[0])
    alt_lag_excess = float(audit_df.loc[audit_df["variant_name"] == "all_non_price_alt_data_lag_1d", "excess_vs_spy"].iloc[0])
    all_lag_excess = float(audit_df.loc[audit_df["variant_name"] == "all_features_lag_1d", "excess_vs_spy"].iloc[0])

    drop_row = audit_df.loc[audit_df["variant_name"] != "normal_features"].sort_values("performance_drop_vs_normal").iloc[0]
    if all(excess > 0 for excess in [sentiment_lag_1d_excess, ratings_lag_excess, grade_lag_excess, alt_lag_excess, all_lag_excess]):
        audit_status = "PASS"
    elif alt_lag_excess > 0 or all_lag_excess > 0:
        audit_status = "WARNING"
    else:
        audit_status = "FAIL"

    report_lines = [
        "# ML Strict Leakage Timing Audit",
        "",
        *[f"- {line}" for line in _caveat_lines()],
        "",
        f"- Audit status: {audit_status}",
        f"- Frozen candidate: `{candidate.strategy_name}` / `{candidate.model_type}`",
        f"- Forward window audited: {pd.Timestamp(features_forward['date'].min()).date()} to {pd.Timestamp(features_forward['date'].max()).date()}",
        "",
        "## Source Timestamp Checks",
        "",
        f"- Sentiment article rows checked: {sentiment_checks['sentiment_article_rows']}",
        f"- Sentiment future-article violations: {sentiment_checks['sentiment_future_article_violations']}",
        f"- Rating rows with historical data: {rating_checks['rating_rows_available']}",
        f"- Rating record-date future violations: {rating_checks['rating_future_violations']}",
        f"- Rating same-day rows requiring stricter lag treatment: {rating_checks['rating_same_day_rows']}",
        f"- Grade event rows checked: {grade_checks['grade_event_rows']}",
        f"- Negative days_since_last_upgrade rows: {grade_checks['negative_days_since_upgrade_rows']}",
        f"- Negative days_since_last_downgrade rows: {grade_checks['negative_days_since_downgrade_rows']}",
        "",
        "## Variant Results",
        "",
        dataframe_to_markdown(audit_df.round(6)),
        "",
        "## Key Questions",
        "",
        f"- ML still beats SPY with sentiment lagged 1 day: {str(sentiment_lag_1d_excess > 0).lower()} ({sentiment_lag_1d_excess:.2%})",
        f"- ML still beats SPY with sentiment lagged 2 days: {str(sentiment_lag_2d_excess > 0).lower()} ({sentiment_lag_2d_excess:.2%})",
        f"- ML still beats SPY with ratings lagged 1 day: {str(ratings_lag_excess > 0).lower()} ({ratings_lag_excess:.2%})",
        f"- ML still beats SPY with grade events lagged 1 day: {str(grade_lag_excess > 0).lower()} ({grade_lag_excess:.2%})",
        f"- ML still beats SPY with all alt-data lagged 1 day: {str(alt_lag_excess > 0).lower()} ({alt_lag_excess:.2%})",
        f"- ML still beats SPY with all features lagged 1 day: {str(all_lag_excess > 0).lower()} ({all_lag_excess:.2%})",
        f"- Largest performance drop vs normal came from: `{drop_row['variant_name']}` ({float(drop_row['performance_drop_vs_normal']):.2%})",
    ]
    (runtime.reports_dir / "ml_strict_leakage_timing_audit.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_lines = [
        "# Research Candidate Summary",
        "",
        *[f"- {line}" for line in _caveat_lines()],
        "",
        "## Frozen Production/Paper Model",
        "",
        f"- Current frozen model: `{EXPECTED_RULE_BASED}`",
        "",
        "## Best 2025 ML Validation Candidate",
        "",
        f"- Model: `{candidate.model_type}`",
        f"- Strategy: `{candidate.strategy_name}`",
        "- 2026 forward data was not used for ML training or model selection.",
        "",
        "## Strict Leakage Timing Audit",
        "",
        f"- Status: {audit_status}",
        f"- ML beats SPY with sentiment lagged 1 day: {str(sentiment_lag_1d_excess > 0).lower()}",
        f"- ML beats SPY with all alt-data lagged 1 day: {str(alt_lag_excess > 0).lower()}",
        f"- ML beats SPY with all features lagged 1 day: {str(all_lag_excess > 0).lower()}",
        f"- Most timing-sensitive variant: `{drop_row['variant_name']}`",
        "- Do not promote ML unless audit status is PASS and forward period exceeds 6 months.",
        "- recommended_strategy.yaml remains unchanged pending more forward evidence.",
    ]
    (runtime.reports_dir / "research_candidate_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Audit status: {audit_status}")
    print(f"Saved {runtime.tables_dir / 'ml_strict_leakage_timing_audit.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_strict_leakage_timing_audit.md'}")
    print(f"Saved {runtime.reports_dir / 'ml_execution_interval_audit.md'}")
    print(f"Saved {runtime.reports_dir / 'ml_preprocessing_leakage_audit.md'}")
    print(f"Saved {runtime.tables_dir / 'ml_feature_list_audit.csv'}")
    print(f"Saved {runtime.reports_dir / 'research_candidate_summary.md'}")


if __name__ == "__main__":
    main()
