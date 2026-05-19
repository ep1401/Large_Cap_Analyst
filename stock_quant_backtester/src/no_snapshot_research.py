from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import (
    _append_holding_rows,
    _apply_sector_limit,
    _build_weights,
    select_rebalance_dates,
)
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import get_future_return_columns
from src.utils import load_dataframe


WALK_FORWARD_WINDOWS = [
    ("2024 H1", "2024-01-01", "2024-06-30"),
    ("2024 H2", "2024-07-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
]
FINAL_5D_WEIGHT_COMPONENT_ORDER = [
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
    "negative_news_flag",
    "recent_downgrade_flag",
]
FINAL_5D_GOOD_COMPONENTS = {
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "net_upgrade_score_30d",
    "relative_strength_21d",
    "relevance_weighted_sentiment_7d",
    "sentiment_change_7d_vs_30d",
    "breakout_63d",
}
FINAL_5D_BAD_COMPONENTS = set(FINAL_5D_WEIGHT_COMPONENT_ORDER) - FINAL_5D_GOOD_COMPONENTS
FINAL_5D_BASELINE_RAW_WEIGHTS = {
    "historical_rating_score": 0.25,
    "historical_positive_rating_ratio": 0.15,
    "historical_negative_rating_ratio": -0.15,
    "net_upgrade_score_30d": 0.20,
    "downgrade_count_30d": -0.15,
    "relative_strength_21d": 0.15,
    "relevance_weighted_sentiment_7d": 0.10,
    "sentiment_change_7d_vs_30d": 0.05,
    "volatility_21d": -0.05,
    "breakout_63d": 0.05,
    "negative_news_flag": 0.0,
    "recent_downgrade_flag": 0.0,
}
FINAL_5D_COMPONENT_EXPORT_COLUMNS = {
    "historical_rating_score": "historical_rating_score_z",
    "historical_positive_rating_ratio": "historical_positive_rating_ratio_z",
    "historical_negative_rating_ratio": "historical_negative_rating_ratio_z",
    "net_upgrade_score_30d": "net_upgrade_score_30d_z",
    "downgrade_count_30d": "downgrade_count_30d_z",
    "relative_strength_21d": "relative_strength_21d_z",
    "relevance_weighted_sentiment_7d": "relevance_weighted_sentiment_7d_z",
    "sentiment_change_7d_vs_30d": "sentiment_change_7d_vs_30d_z",
    "volatility_21d": "volatility_21d_z",
    "breakout_63d": "breakout_63d_component",
    "negative_news_flag": "negative_news_component",
    "recent_downgrade_flag": "recent_downgrade_component",
}


@dataclass(slots=True)
class CustomStrategyDefinition:
    name: str
    display_name: str
    score_builder: Callable[[pd.DataFrame], pd.DataFrame]
    require_historical_rating_count: bool = False
    min_historical_rating_count: int = 5
    require_historical_grade_data: bool = False
    exclude_strong_negative_news: bool = False
    exclude_recent_downgrades: bool = False


def _safe_series(df: pd.DataFrame, column: str, default: float | bool | None = np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _cross_sectional_zscore(series: pd.Series, clip_range: tuple[float, float] = (-3.0, 3.0)) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    zscore = (values - values.mean()) / std
    return zscore.clip(*clip_range).fillna(0.0)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2%}"


def fmt_float(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()


def safe_metrics(frame: pd.DataFrame, holding_period_days: int) -> dict[str, float]:
    if frame.empty:
        return {
            "total_return": float("nan"),
            "spy_total_return": float("nan"),
            "excess_total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "average_turnover": float("nan"),
            "average_selected_count": float("nan"),
            "weeks_beating_spy": float("nan"),
            "number_of_rebalance_periods": 0,
            "number_of_invested_periods": 0,
        }
    return calculate_performance_metrics(frame, holding_period_days=holding_period_days)


def get_features_path(config: Config, features_path: str | Path | None = None) -> Path:
    return Path(features_path) if features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"


def load_features(config: Config, features_path: str | Path | None = None) -> pd.DataFrame:
    return load_dataframe(get_features_path(config, features_path), parse_dates=["date"])


def get_best_5d_config(config: Config) -> dict[str, object]:
    defaults: dict[str, object] = {
        "strategy_name": "final_quant_5d_no_snapshot_no_sma_filter",
        "top_n": 10,
        "position_sizing": "equal_weight",
        "max_names_per_sector": None,
        "max_single_name_weight": 0.15,
        "total_cost_bps": float(config.transaction_cost_bps),
        "avoid_strong_negative_news": False,
        "avoid_recent_downgrades": False,
    }
    path = config.tables_dir / "position_sizing_comparison.csv"
    if not path.exists():
        return defaults
    df = load_dataframe(path)
    selected = df.loc[df["selected_best"] == True]  # noqa: E712
    if selected.empty:
        return defaults
    best = selected.iloc[0]
    return {
        "strategy_name": str(best["strategy_name"]),
        "top_n": int(best["top_n"]),
        "position_sizing": str(best["position_sizing"]),
        "max_names_per_sector": None if pd.isna(best["max_names_per_sector"]) else int(best["max_names_per_sector"]),
        "max_single_name_weight": float(best["max_single_name_weight"]),
        "total_cost_bps": float(best["total_cost_bps"]),
        "avoid_strong_negative_news": False,
        "avoid_recent_downgrades": False,
    }


def _base_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "historical_rating_score": _cross_sectional_zscore(_safe_series(df, "historical_rating_score")),
        "historical_positive_ratio": _cross_sectional_zscore(_safe_series(df, "historical_positive_rating_ratio")),
        "historical_negative_ratio": _cross_sectional_zscore(_safe_series(df, "historical_negative_rating_ratio")),
        "net_upgrade_score_30d": _cross_sectional_zscore(_safe_series(df, "net_upgrade_score_30d")),
        "downgrade_count_30d": _cross_sectional_zscore(_safe_series(df, "downgrade_count_30d")),
        "relative_strength_21d": _cross_sectional_zscore(_safe_series(df, "relative_strength_21d")),
        "relative_strength_63d": _cross_sectional_zscore(_safe_series(df, "relative_strength_63d")),
        "relevance_weighted_sentiment_7d": _cross_sectional_zscore(_safe_series(df, "relevance_weighted_sentiment_7d")),
        "sentiment_change_7d_vs_30d": _cross_sectional_zscore(_safe_series(df, "sentiment_change_7d_vs_30d")),
        "negative_news_ratio_7d": _cross_sectional_zscore(_safe_series(df, "negative_news_ratio_7d")),
        "volatility_21d": _cross_sectional_zscore(_safe_series(df, "volatility_21d")),
        "distance_to_63d_high": _cross_sectional_zscore(-_safe_series(df, "distance_to_63d_high")),
        "above_sma_50": _cross_sectional_zscore(_safe_series(df, "above_sma_50")),
        "above_sma_200": _cross_sectional_zscore(_safe_series(df, "above_sma_200")),
        "breakout_63d": _safe_series(df, "breakout_63d", False).fillna(False).astype(float),
        "strong_negative_news_flag": _safe_series(df, "strong_negative_news_flag", False).fillna(False).astype(float),
        "recent_downgrade_flag_30d": _safe_series(df, "recent_downgrade_flag_30d", False).fillna(False).astype(float),
    }


def normalize_final_5d_weights(weights: dict[str, float], max_abs_weight: float = 0.35) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for component in FINAL_5D_WEIGHT_COMPONENT_ORDER:
        value = float(weights.get(component, 0.0))
        if component in FINAL_5D_GOOD_COMPONENTS and value < 0:
            raise ValueError(f"{component} must have a non-negative weight.")
        if component in FINAL_5D_BAD_COMPONENTS and value > 0:
            raise ValueError(f"{component} must have a non-positive weight.")
        normalized[component] = value

    abs_sum = sum(abs(value) for value in normalized.values())
    if abs_sum <= 0:
        raise ValueError("At least one 5D component weight must be non-zero.")
    normalized = {component: value / abs_sum for component, value in normalized.items()}
    largest_abs = max(abs(value) for value in normalized.values())
    if largest_abs > max_abs_weight + 1e-9:
        raise ValueError(f"Normalized 5D weights exceed max_abs_weight={max_abs_weight}: {largest_abs:.4f}")
    return normalized


def get_baseline_final_5d_weights() -> dict[str, float]:
    return normalize_final_5d_weights(FINAL_5D_BASELINE_RAW_WEIGHTS)


def final_5d_weight_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    base = _base_components(df)
    return {
        "historical_rating_score": base["historical_rating_score"],
        "historical_positive_rating_ratio": base["historical_positive_ratio"],
        "historical_negative_rating_ratio": base["historical_negative_ratio"],
        "net_upgrade_score_30d": base["net_upgrade_score_30d"],
        "downgrade_count_30d": base["downgrade_count_30d"],
        "relative_strength_21d": base["relative_strength_21d"],
        "relevance_weighted_sentiment_7d": base["relevance_weighted_sentiment_7d"],
        "sentiment_change_7d_vs_30d": base["sentiment_change_7d_vs_30d"],
        "volatility_21d": base["volatility_21d"],
        "breakout_63d": base["breakout_63d"],
        "negative_news_flag": base["strong_negative_news_flag"],
        "recent_downgrade_flag": base["recent_downgrade_flag_30d"],
    }


def score_final_quant_5d_with_weights(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    normalized = normalize_final_5d_weights(weights)
    components = final_5d_weight_components(df)
    scored = df.copy()
    score = pd.Series(0.0, index=scored.index, dtype=float)
    for component_name in FINAL_5D_WEIGHT_COMPONENT_ORDER:
        score = score + normalized[component_name] * components[component_name]
    scored["score"] = score
    return scored


def build_final_5d_component_frame(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    active_weights = get_baseline_final_5d_weights() if weights is None else normalize_final_5d_weights(weights)
    components = final_5d_weight_components(df)
    component_frame = df.copy()
    score = pd.Series(0.0, index=component_frame.index, dtype=float)
    for component_name, output_column in FINAL_5D_COMPONENT_EXPORT_COLUMNS.items():
        component_frame[output_column] = components[component_name]
        component_frame[f"weight_{component_name}"] = active_weights[component_name]
        score = score + active_weights[component_name] * components[component_name]
    component_frame["score"] = score
    return component_frame


def score_final_quant_5d(df: pd.DataFrame) -> pd.DataFrame:
    return score_final_quant_5d_with_weights(df, get_baseline_final_5d_weights())


def score_final_quant_5d_simplified(df: pd.DataFrame) -> pd.DataFrame:
    components = _base_components(df)
    scored = df.copy()
    scored["score"] = (
        0.25 * components["historical_rating_score"]
        + 0.20 * components["net_upgrade_score_30d"]
        - 0.20 * components["downgrade_count_30d"]
        + 0.20 * components["relative_strength_21d"]
        + 0.10 * components["relevance_weighted_sentiment_7d"]
        - 0.10 * components["negative_news_ratio_7d"]
        - 0.05 * components["volatility_21d"]
    )
    return scored


def score_ablation_variant(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    components = _base_components(df)
    scored = df.copy()

    historical_ratings = (
        0.25 * components["historical_rating_score"]
        + 0.15 * components["historical_positive_ratio"]
        - 0.15 * components["historical_negative_ratio"]
    )
    grade_events = 0.20 * components["net_upgrade_score_30d"] - 0.15 * components["downgrade_count_30d"]
    sentiment = 0.10 * components["relevance_weighted_sentiment_7d"] + 0.05 * components["sentiment_change_7d_vs_30d"]
    relative_strength = 0.15 * components["relative_strength_21d"]
    volatility_penalty = -0.05 * components["volatility_21d"]
    breakout = 0.05 * components["breakout_63d"]

    variant_scores = {
        "full_model": historical_ratings + grade_events + sentiment + relative_strength + volatility_penalty + breakout,
        "remove_historical_rating_score": grade_events + sentiment + relative_strength + volatility_penalty + breakout,
        "remove_grade_events": historical_ratings + sentiment + relative_strength + volatility_penalty + breakout,
        "remove_sentiment": historical_ratings + grade_events + relative_strength + volatility_penalty + breakout,
        "remove_relative_strength": historical_ratings + grade_events + sentiment + volatility_penalty + breakout,
        "remove_volatility_penalty": historical_ratings + grade_events + sentiment + relative_strength + breakout,
        "remove_breakout": historical_ratings + grade_events + sentiment + relative_strength + volatility_penalty,
        "remove_negative_news_filter": historical_ratings + grade_events + sentiment + relative_strength + volatility_penalty + breakout,
        "remove_recent_downgrade_filter": historical_ratings + grade_events + sentiment + relative_strength + volatility_penalty + breakout,
        "only_historical_ratings_and_events": historical_ratings + grade_events,
        "only_technical_and_sentiment": sentiment + relative_strength + volatility_penalty + breakout,
        "only_historical_ratings_and_relative_strength": historical_ratings + relative_strength,
    }
    if variant_name not in variant_scores:
        raise ValueError(f"Unsupported ablation variant: {variant_name}")
    scored["score"] = variant_scores[variant_name]
    return scored


def build_final_quant_5d_definition() -> CustomStrategyDefinition:
    return CustomStrategyDefinition(
        name="final_quant_5d_no_snapshot_no_sma_filter",
        display_name="Final Quant 5D - No SMA Filter",
        score_builder=score_final_quant_5d,
        require_historical_rating_count=True,
        min_historical_rating_count=5,
        require_historical_grade_data=True,
        exclude_strong_negative_news=True,
        exclude_recent_downgrades=True,
    )


def build_weight_tuned_final_quant_5d_definition(weights: dict[str, float]) -> CustomStrategyDefinition:
    normalized = normalize_final_5d_weights(weights)
    return CustomStrategyDefinition(
        name="final_quant_5d_weight_tuned_no_snapshot",
        display_name="Final Quant 5D - Weight Tuned No Snapshot",
        score_builder=lambda df, local_weights=normalized: score_final_quant_5d_with_weights(df, local_weights),
        require_historical_rating_count=True,
        min_historical_rating_count=5,
        require_historical_grade_data=True,
        exclude_strong_negative_news=False,
        exclude_recent_downgrades=False,
    )


def build_simplified_5d_definition(
    exclude_strong_negative_news: bool = True,
    exclude_recent_downgrades: bool = False,
) -> CustomStrategyDefinition:
    return CustomStrategyDefinition(
        name="final_quant_5d_simplified_no_snapshot",
        display_name="Final Quant 5D Simplified - No Snapshot",
        score_builder=score_final_quant_5d_simplified,
        require_historical_rating_count=True,
        min_historical_rating_count=5,
        require_historical_grade_data=True,
        exclude_strong_negative_news=exclude_strong_negative_news,
        exclude_recent_downgrades=exclude_recent_downgrades,
    )


def build_ablation_definition(variant_name: str) -> CustomStrategyDefinition:
    exclude_negative_news = variant_name != "remove_negative_news_filter"
    exclude_recent_downgrades = variant_name != "remove_recent_downgrade_filter"
    return CustomStrategyDefinition(
        name=variant_name,
        display_name=variant_name,
        score_builder=lambda df, name=variant_name: score_ablation_variant(df, name),
        require_historical_rating_count=True,
        min_historical_rating_count=5,
        require_historical_grade_data=True,
        exclude_strong_negative_news=exclude_negative_news,
        exclude_recent_downgrades=exclude_recent_downgrades,
    )


def build_eligible_universe(
    day_slice: pd.DataFrame,
    holding_period_days: int,
    benchmark: str,
    min_avg_dollar_volume: float,
    require_historical_rating_count: bool = False,
    min_historical_rating_count: int = 5,
    require_historical_grade_data: bool = False,
    exclude_strong_negative_news: bool = False,
    exclude_recent_downgrades: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    future_return_column, _, _ = get_future_return_columns(holding_period_days)
    eligible = day_slice.copy()
    mask = pd.Series(True, index=eligible.index)
    mask &= eligible["ticker"].ne(benchmark)
    mask &= _safe_series(eligible, "adjusted_close").notna()
    mask &= _safe_series(eligible, future_return_column).notna()
    mask &= _safe_series(eligible, "avg_dollar_volume_21d", 0).fillna(0) >= min_avg_dollar_volume
    diagnostics = {
        "starting_universe_count": int(len(eligible)),
        "passed_liquidity_count": int(mask.sum()),
    }
    if require_historical_rating_count:
        mask &= _safe_series(eligible, "historical_rating_count_data_available", False).fillna(False)
        diagnostics["passed_historical_rating_count_data_count"] = int(mask.sum())
        mask &= _safe_series(eligible, "historical_total_ratings", 0).fillna(0) >= min_historical_rating_count
        diagnostics["passed_min_historical_rating_count_count"] = int(mask.sum())
    if require_historical_grade_data:
        mask &= _safe_series(eligible, "historical_grade_data_available", False).fillna(False)
        diagnostics["passed_historical_grade_data_count"] = int(mask.sum())
    if exclude_strong_negative_news:
        mask &= ~_safe_series(eligible, "strong_negative_news_flag", False).fillna(False)
        diagnostics["passed_negative_news_filter_count"] = int(mask.sum())
    if exclude_recent_downgrades:
        mask &= ~_safe_series(eligible, "recent_downgrade_flag_30d", False).fillna(False)
        diagnostics["passed_recent_downgrade_filter_count"] = int(mask.sum())
    qualified = eligible.loc[mask].copy()
    diagnostics["final_pass_count"] = int(len(qualified))
    return qualified, diagnostics


def run_custom_weekly_backtest(
    features: pd.DataFrame,
    definition: CustomStrategyDefinition,
    holding_period_days: int,
    benchmark: str,
    top_n: int | None,
    transaction_cost_bps: float,
    min_avg_dollar_volume: float,
    max_names_per_sector: int | None = None,
    position_sizing: str = "equal_weight",
    max_single_name_weight: float = 0.15,
    score_threshold: float | None = None,
    top_percentile: float | None = None,
    allow_cash_if_threshold_unmet: bool = False,
    rebalance_dates: list[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(holding_period_days)
    if rebalance_dates is None:
        rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)

    weekly_rows: list[dict] = []
    holding_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    portfolio_value = 10000.0
    spy_value = 10000.0
    previous_weights: dict[str, float] = {}

    for rebalance_date in rebalance_dates:
        day_all = df.loc[df["date"] == rebalance_date].copy()
        day_slice = day_all.loc[day_all["ticker"] != benchmark].copy()
        qualified, diagnostics = build_eligible_universe(
            day_slice=day_slice,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
            min_avg_dollar_volume=min_avg_dollar_volume,
            require_historical_rating_count=definition.require_historical_rating_count,
            min_historical_rating_count=definition.min_historical_rating_count,
            require_historical_grade_data=definition.require_historical_grade_data,
            exclude_strong_negative_news=definition.exclude_strong_negative_news,
            exclude_recent_downgrades=definition.exclude_recent_downgrades,
        )
        scored = definition.score_builder(qualified).sort_values("score", ascending=False).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)

        passed = scored.copy()
        if score_threshold is not None:
            passed = passed.loc[pd.to_numeric(passed["score"], errors="coerce").fillna(-np.inf) > score_threshold].copy()
        if top_percentile is not None and not passed.empty:
            percentile_cutoff = pd.to_numeric(scored["score"], errors="coerce").quantile(1 - top_percentile)
            passed = passed.loc[pd.to_numeric(passed["score"], errors="coerce").fillna(-np.inf) >= percentile_cutoff].copy()

        if top_n is None:
            selected = passed.copy()
        else:
            selected = passed.head(top_n).copy()
            if not allow_cash_if_threshold_unmet and len(selected) < top_n:
                refill = scored.loc[~scored["ticker"].isin(selected["ticker"])].head(top_n - len(selected))
                selected = pd.concat([selected, refill], ignore_index=True)
                selected = selected.sort_values("score", ascending=False).head(top_n).copy()

        selected = _apply_sector_limit(selected, max_names_per_sector=max_names_per_sector)
        selected = _build_weights(
            selected,
            exposure=1.0 if not selected.empty else 0.0,
            use_inverse_vol_weighting=position_sizing == "inverse_volatility",
            max_single_name_weight=max_single_name_weight,
            position_sizing=position_sizing,
        )
        selected_count = len(selected)
        new_weights = dict(zip(selected["ticker"], selected["weight"])) if selected_count else {}
        turnover = sum(abs(new_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in set(new_weights) | set(previous_weights))
        gross_return = float((selected["weight"] * selected[future_return_column].fillna(0.0)).sum()) if selected_count else 0.0
        transaction_cost = turnover * float(transaction_cost_bps) / 10000.0
        net_return = gross_return - transaction_cost

        benchmark_slice = day_all.loc[day_all["ticker"] == benchmark, future_spy_return_column]
        spy_return = float(benchmark_slice.iloc[0]) if not benchmark_slice.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        diagnostics_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": definition.name,
                **diagnostics,
                "threshold_pass_count": int(len(passed)),
                "selected_count": selected_count,
                "average_score_selected": float(selected["score"].mean()) if selected_count else float("nan"),
            }
        )
        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": definition.name,
                "holding_period_days": holding_period_days,
                "top_n": top_n if top_n is not None else len(selected),
                "selected_count": selected_count,
                "qualified_count": len(qualified),
                "threshold_pass_count": int(len(passed)),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": excess_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": float(selected["weight"].sum()) if selected_count else 0.0,
                "position_sizing": position_sizing,
                "allow_cash_if_threshold_unmet": allow_cash_if_threshold_unmet,
                "score_threshold": score_threshold,
                "top_percentile": top_percentile,
            }
        )
        _append_holding_rows(
            holding_rows,
            selected,
            rebalance_date,
            definition.name,
            holding_period_days,
            future_return_column,
            future_excess_return_column,
        )
        previous_weights = new_weights

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows), pd.DataFrame(diagnostics_rows)


def run_benchmark_buy_hold(
    features: pd.DataFrame,
    holding_period_days: int,
    benchmark_ticker: str,
    benchmark: str,
) -> pd.DataFrame:
    future_return_column, future_spy_return_column, _ = get_future_return_columns(holding_period_days)
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)
    weekly_rows: list[dict] = []
    portfolio_value = 10000.0
    spy_value = 10000.0
    for rebalance_date in rebalance_dates:
        day = df.loc[(df["date"] == rebalance_date) & (df["ticker"] == benchmark_ticker)].copy()
        benchmark_day = df.loc[(df["date"] == rebalance_date) & (df["ticker"] == benchmark)].copy()
        if day.empty:
            continue
        net_return = float(day[future_return_column].iloc[0])
        spy_return = float(benchmark_day[future_spy_return_column].iloc[0]) if not benchmark_day.empty else 0.0
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return
        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": benchmark_ticker.lower(),
                "holding_period_days": holding_period_days,
                "top_n": 1,
                "selected_count": 1,
                "qualified_count": 1,
                "threshold_pass_count": 1,
                "gross_return": net_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": 1.0,
            }
        )
    return pd.DataFrame(weekly_rows)


def summarize_backtest(
    weekly: pd.DataFrame,
    holding_period_days: int,
    label: str,
) -> dict[str, object]:
    metrics = safe_metrics(weekly, holding_period_days=holding_period_days)
    summary: dict[str, object] = {
        "label": label,
        "holding_period_days": holding_period_days,
        "full_period_total_return": metrics["total_return"],
        "full_period_excess_return_vs_spy": metrics["excess_total_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "average_turnover": metrics["average_turnover"],
        "average_holdings": metrics["average_selected_count"],
        "percent_periods_invested": (
            float((weekly["exposure"] > 0).mean()) if not weekly.empty and "exposure" in weekly.columns else float("nan")
        ),
        "number_of_rebalance_periods": metrics["number_of_rebalance_periods"],
    }
    for window_name, start, end in WALK_FORWARD_WINDOWS:
        window_metrics = safe_metrics(slice_period(weekly, start, end), holding_period_days=holding_period_days)
        summary[f"{window_name.lower().replace(' ', '_')}_excess_return_vs_spy"] = window_metrics["excess_total_return"]
        summary[f"{window_name.lower().replace(' ', '_')}_sharpe_ratio"] = window_metrics["sharpe_ratio"]
    summary["windows_beating_spy"] = sum(
        int(bool(pd.notna(summary[f"{window_name.lower().replace(' ', '_')}_excess_return_vs_spy"]) and summary[f"{window_name.lower().replace(' ', '_')}_excess_return_vs_spy"] > 0))
        for window_name, _, _ in WALK_FORWARD_WINDOWS
    )
    return summary
