from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.no_snapshot_research import (
    FINAL_5D_WEIGHT_COMPONENT_ORDER,
    build_eligible_universe,
    build_final_quant_5d_definition,
    build_weight_tuned_final_quant_5d_definition,
    final_5d_weight_components,
    run_custom_weekly_backtest,
    select_rebalance_dates,
)
from src.promoted_weights import load_promoted_final_5d_tuned_weights
from src.scoring import get_future_return_columns
from src.scoring import strategy_display_name
from src.utils import load_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical."
SNAPSHOT_CAVEAT = "Snapshot analyst target models are excluded from the main historically safer ranking."
HISTORICAL_NOTE = "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date."
SENTIMENT_CAVEAT = "News sentiment depends on Alpha Vantage coverage and classification."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
REGIME_CAVEAT = "Regime filters were tested and are not recommended for the main model based on current results."
RESEARCH_CAVEAT = "This is research/paper trading only, not financial advice."
COST_SENSITIVITY_CAVEAT = "The current model is cost-sensitive and should remain paper-trading only unless it survives realistic cost assumptions."
SIGNAL_GROUP_COMPONENTS = {
    "historical_rating_counts": [
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
    ],
    "historical_grade_events": [
        "net_upgrade_score_30d",
        "downgrade_count_30d",
    ],
    "technical_momentum": [
        "relative_strength_21d",
        "breakout_63d",
    ],
    "sentiment": [
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
    ],
    "risk_penalties": [
        "volatility_21d",
        "negative_news_flag",
        "recent_downgrade_flag",
    ],
}


@dataclass(slots=True)
class RecommendedStrategyConfig:
    strategy_name: str
    holding_period_days: int
    top_n: int
    position_sizing: str
    allow_cash: bool
    threshold: float | None
    regime_filter: str
    long_short: bool
    total_cost_bps: float
    enter_rank: int | None = None
    hold_rank: int | None = None
    max_holding_days: int | None = None
    rebalance_frequency_days: int | None = None
    max_turnover_per_rebalance: float | None = None


def caveat_lines() -> list[str]:
    return [
        BACKTEST_CAVEAT,
        SNAPSHOT_CAVEAT,
        HISTORICAL_NOTE,
        SENTIMENT_CAVEAT,
        LONG_SHORT_CAVEAT,
        REGIME_CAVEAT,
        COST_SENSITIVITY_CAVEAT,
        RESEARCH_CAVEAT,
    ]


def config_path(project_root: Path) -> Path:
    return project_root / "configs" / "recommended_strategy.yaml"


def default_recommended_strategy_config() -> RecommendedStrategyConfig:
    return RecommendedStrategyConfig(
        strategy_name="final_quant_5d_weight_tuned_no_snapshot",
        holding_period_days=5,
        top_n=10,
        position_sizing="equal_weight",
        allow_cash=False,
        threshold=None,
        regime_filter="none",
        long_short=False,
        total_cost_bps=10.0,
        enter_rank=None,
        hold_rank=None,
        max_holding_days=None,
        rebalance_frequency_days=None,
        max_turnover_per_rebalance=None,
    )


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


def load_recommended_strategy_config(project_root: Path | None = None) -> RecommendedStrategyConfig:
    root = project_root or Path(__file__).resolve().parents[1]
    path = config_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing recommended strategy config: {path}")
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return RecommendedStrategyConfig(**data)


def save_recommended_strategy_config(config: RecommendedStrategyConfig, project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in asdict(config).items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str):
            rendered = f'"{value}"'
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_promoted_tuned_weights(project_root: Path | None = None) -> dict[str, float]:
    return load_promoted_final_5d_tuned_weights(project_root)


def build_strategy_definition_from_config(recommended: RecommendedStrategyConfig, project_root: Path | None = None):
    if recommended.strategy_name in {
        "final_quant_5d_weight_tuned_no_snapshot",
        "final_quant_5d_weight_tuned_low_turnover_no_snapshot",
    }:
        return build_weight_tuned_final_quant_5d_definition(load_promoted_tuned_weights(project_root))
    if recommended.strategy_name == "final_quant_5d_no_snapshot_no_sma_filter":
        return build_final_quant_5d_definition()
    raise ValueError(f"Unsupported recommended strategy for custom execution: {recommended.strategy_name}")


def load_runtime_and_recommended(
    features_path: str | Path | None = None,
) -> tuple[Config, RecommendedStrategyConfig, pd.DataFrame]:
    runtime = Config.from_env()
    recommended = load_recommended_strategy_config(runtime.project_root)
    features = load_dataframe(
        Path(features_path) if features_path else runtime.final_dir / "features_panel_2023-01-01_2026-01-01.csv",
        parse_dates=["date"],
    )
    return runtime, recommended, features


def _weighted_component_contributions(day_slice: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    weights = load_promoted_tuned_weights(project_root)
    components = final_5d_weight_components(day_slice)
    out = day_slice.copy()
    for component, series in components.items():
        out[f"{component}_component"] = series
        out[f"{component}_contribution"] = weights[component] * series
    contribution_columns = [f"{component}_contribution" for component in FINAL_5D_WEIGHT_COMPONENT_ORDER]
    out["score"] = out[contribution_columns].sum(axis=1)
    return out


def latest_recommended_holdings(
    features: pd.DataFrame,
    runtime: Config,
    recommended: RecommendedStrategyConfig,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    definition = build_strategy_definition_from_config(recommended, runtime.project_root)
    latest_date = pd.Timestamp(features["date"].max())
    selected = pd.DataFrame()
    selected_date = latest_date
    for candidate_date in sorted(pd.to_datetime(features["date"]).drop_duplicates(), reverse=True):
        day = features.loc[(features["date"] == candidate_date) & (features["ticker"] != runtime.benchmark)].copy()
        mask = pd.Series(True, index=day.index)
        mask &= day["adjusted_close"].notna()
        mask &= pd.to_numeric(day["avg_dollar_volume_21d"], errors="coerce").fillna(0.0) >= runtime.min_avg_dollar_volume
        if definition.require_historical_rating_count:
            mask &= day["historical_rating_count_data_available"].fillna(False).astype(bool)
            mask &= pd.to_numeric(day["historical_total_ratings"], errors="coerce").fillna(0.0) >= definition.min_historical_rating_count
        if definition.require_historical_grade_data:
            mask &= day["historical_grade_data_available"].fillna(False).astype(bool)
        if definition.exclude_strong_negative_news:
            mask &= ~day["strong_negative_news_flag"].fillna(False).astype(bool)
        if definition.exclude_recent_downgrades:
            mask &= ~day["recent_downgrade_flag_30d"].fillna(False).astype(bool)
        qualified = day.loc[mask].copy()
        if qualified.empty:
            continue
        if recommended.strategy_name == "final_quant_5d_weight_tuned_no_snapshot":
            scored = _weighted_component_contributions(qualified, runtime.project_root).sort_values("score", ascending=False)
        else:
            scored = definition.score_builder(qualified).sort_values("score", ascending=False)
        selected = scored.head(recommended.top_n).copy()
        if len(selected) >= recommended.top_n:
            selected_date = pd.Timestamp(candidate_date)
            break
    if selected.empty:
        return selected, selected_date
    selected = selected.copy()
    selected["rank"] = np.arange(1, len(selected) + 1, dtype=float)
    selected["weight"] = 1.0 / len(selected)
    selected["strategy_name"] = recommended.strategy_name
    selected["holding_period_days"] = recommended.holding_period_days
    selected["position_sizing"] = recommended.position_sizing
    selected["total_cost_bps"] = recommended.total_cost_bps
    selected["min_score_threshold"] = recommended.threshold
    selected["allow_cash"] = recommended.allow_cash
    selected["cash_weight"] = 0.0
    selected["date"] = selected_date
    return selected, selected_date


def run_recommended_backtest(
    features: pd.DataFrame,
    runtime: Config,
    recommended: RecommendedStrategyConfig,
    top_n: int | None = None,
    total_cost_bps: float | None = None,
    position_sizing: str | None = None,
    rebalance_dates: list[pd.Timestamp] | None = None,
    definition_override=None,
):
    definition = definition_override or build_strategy_definition_from_config(recommended, runtime.project_root)
    return run_custom_weekly_backtest(
        features=features,
        definition=definition,
        holding_period_days=recommended.holding_period_days,
        benchmark=runtime.benchmark,
        top_n=recommended.top_n if top_n is None else top_n,
        transaction_cost_bps=float(recommended.total_cost_bps if total_cost_bps is None else total_cost_bps),
        min_avg_dollar_volume=runtime.min_avg_dollar_volume,
        max_names_per_sector=None,
        position_sizing=recommended.position_sizing if position_sizing is None else position_sizing,
        score_threshold=recommended.threshold,
        allow_cash_if_threshold_unmet=recommended.allow_cash,
        rebalance_dates=rebalance_dates,
    )


def precompute_recommended_panels(
    features: pd.DataFrame,
    runtime: Config,
    recommended: RecommendedStrategyConfig,
) -> list[tuple[pd.Timestamp, pd.DataFrame, float]]:
    future_return_column, future_spy_return_column, _ = get_future_return_columns(recommended.holding_period_days)
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float]] = []
    definition = build_strategy_definition_from_config(recommended, runtime.project_root)
    rebalance_dates = select_rebalance_dates(features, recommended.holding_period_days, runtime.benchmark)
    for rebalance_date in rebalance_dates:
        day_all = features.loc[features["date"] == rebalance_date].copy()
        benchmark_slice = day_all.loc[day_all["ticker"] == runtime.benchmark, future_spy_return_column]
        if benchmark_slice.empty or pd.isna(benchmark_slice.iloc[0]):
            continue
        day = day_all.loc[day_all["ticker"] != runtime.benchmark].copy()
        qualified, _ = build_eligible_universe(
            day_slice=day,
            holding_period_days=recommended.holding_period_days,
            benchmark=runtime.benchmark,
            min_avg_dollar_volume=runtime.min_avg_dollar_volume,
            require_historical_rating_count=definition.require_historical_rating_count,
            min_historical_rating_count=definition.min_historical_rating_count,
            require_historical_grade_data=definition.require_historical_grade_data,
            exclude_strong_negative_news=definition.exclude_strong_negative_news,
            exclude_recent_downgrades=definition.exclude_recent_downgrades,
        )
        if qualified.empty:
            continue
        if recommended.strategy_name == "final_quant_5d_weight_tuned_no_snapshot":
            scored = _weighted_component_contributions(qualified, runtime.project_root)
        else:
            scored = definition.score_builder(qualified)
        panel = scored.copy()
        panel["base_score"] = panel["score"]
        panel["future_return_used"] = pd.to_numeric(panel[future_return_column], errors="coerce").fillna(0.0)
        panels.append((pd.Timestamp(rebalance_date), panel, float(benchmark_slice.iloc[0])))
    return panels


def precompute_recommended_low_turnover_panels(
    features: pd.DataFrame,
    runtime: Config,
    recommended: RecommendedStrategyConfig,
) -> list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]]:
    future_return_column, _, _ = get_future_return_columns(recommended.holding_period_days)
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]] = []
    rebalance_frequency_days = recommended.rebalance_frequency_days or recommended.holding_period_days
    rebalance_dates = select_rebalance_dates(
        features,
        holding_period_days=recommended.holding_period_days,
        benchmark=runtime.benchmark,
        rebalance_frequency_days=rebalance_frequency_days,
    )
    available_dates = sorted(pd.to_datetime(features["date"]).drop_duplicates())
    if available_dates and rebalance_dates and pd.Timestamp(rebalance_dates[-1]) < pd.Timestamp(available_dates[-1]):
        rebalance_dates = [*rebalance_dates, pd.Timestamp(available_dates[-1])]
    for rebalance_date in rebalance_dates:
        day_all = features.loc[features["date"] == rebalance_date].copy()
        benchmark_slice = day_all.loc[day_all["ticker"] == runtime.benchmark, "adjusted_close"]
        if benchmark_slice.empty or pd.isna(benchmark_slice.iloc[0]):
            continue
        day = day_all.loc[day_all["ticker"] != runtime.benchmark].copy()
        mask = pd.Series(True, index=day.index)
        mask &= day["adjusted_close"].notna()
        mask &= pd.to_numeric(day["avg_dollar_volume_21d"], errors="coerce").fillna(0.0) >= runtime.min_avg_dollar_volume
        mask &= day["historical_rating_count_data_available"].fillna(False).astype(bool)
        mask &= pd.to_numeric(day["historical_total_ratings"], errors="coerce").fillna(0.0) >= 5
        mask &= day["historical_grade_data_available"].fillna(False).astype(bool)
        qualified = day.loc[mask].copy()
        if qualified.empty:
            continue
        scored = _weighted_component_contributions(qualified, runtime.project_root)
        scored["future_return_used"] = pd.to_numeric(scored[future_return_column], errors="coerce").fillna(0.0)
        scored["strong_negative_news_flag"] = scored["strong_negative_news_flag"].fillna(False).astype(bool)
        scored["recent_downgrade_flag_30d"] = scored["recent_downgrade_flag_30d"].fillna(False).astype(bool)
        price_lookup = day.loc[:, ["ticker", "adjusted_close"]].copy()
        panels.append((pd.Timestamp(rebalance_date), scored.copy(), float(benchmark_slice.iloc[0]), price_lookup))
    return panels


def run_low_turnover_recommended_backtest(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    top_n: int,
    cost_bps: float,
    enter_rank: int,
    hold_rank: int,
    max_holding_days: int,
    rebalance_frequency_days: int,
    strategy_name: str,
    max_turnover_per_rebalance: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    portfolio_value = 10000.0
    spy_value = 10000.0

    if len(panels) < 2:
        return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows), pd.DataFrame(action_rows)

    for idx in range(len(panels) - 1):
        rebalance_date, panel, spy_price, _ = panels[idx]
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
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "SELL", "reason": reason})
            holdings.pop(ticker, None)

        current_after_forced = list(holdings.keys())
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_after_forced
        ]
        turnover_used = len(forced_sells) * (1.0 / top_n)
        turnover_budget = float("inf") if max_turnover_per_rebalance is None else max(0.0, float(max_turnover_per_rebalance) - turnover_used)
        allowed_sells = len(discretionary_sells)
        allowed_buys = len(desired_buys)
        if max_turnover_per_rebalance is not None:
            unit = 1.0 / top_n
            max_steps = int(turnover_budget / unit + 1e-9)
            allowed_sells = min(len(discretionary_sells), max_steps)
            remaining_steps = max(0, max_steps - allowed_sells)
            allowed_buys = min(len(desired_buys), remaining_steps)

        discretionary_sells = sorted(discretionary_sells, key=lambda item: item[1], reverse=True)
        executed_sells = discretionary_sells[:allowed_sells]
        for ticker, rank in executed_sells:
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "SELL", "reason": f"rank>{hold_rank}:{rank}"})
            holdings.pop(ticker, None)

        current_tickers = list(holdings.keys())
        open_slots = max(0, top_n - len(current_tickers))
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_tickers
        ]
        executed_buys = desired_buys[: min(open_slots, allowed_buys)]
        for ticker in executed_buys:
            row = ranked_by_ticker.loc[ticker]
            holdings[ticker] = {
                "holding_days": 0,
                "entry_date": rebalance_date,
                "action": "BUY",
            }
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "BUY", "reason": f"enter_rank<={enter_rank}"})

        selected_tickers = [ticker for ticker in ranked["ticker"].tolist() if ticker in holdings][:top_n]
        for ticker in list(holdings):
            if ticker not in selected_tickers:
                holdings.pop(ticker, None)

        selected = ranked.loc[ranked["ticker"].isin(selected_tickers)].copy().sort_values("rank")
        selected_count = len(selected)
        if selected_count > 0:
            weight = 1.0 / selected_count
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
                    realized_return = float(row["future_return_used"])
                gross_return += float(row["weight"]) * realized_return
        net_return = gross_return - transaction_cost
        if next_spy_price is not None and spy_price:
            spy_return = next_spy_price / float(spy_price) - 1
        else:
            spy_return = 0.0
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        for ticker in list(holdings):
            row = ranked_by_ticker.loc[ticker]
            action_type = "HOLD"
            if any(action["date"] == rebalance_date and action["ticker"] == ticker and action["action"] == "BUY" for action in action_rows):
                action_type = "BUY"
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)
            holdings[ticker]["weight"] = float(new_weights.get(ticker, 0.0))
            holdings[ticker]["last_rank"] = int(row["rank"])
            holdings[ticker]["last_score"] = float(row["score"])
            holdings[ticker]["last_action"] = action_type
            holding_rows.append(
                {
                    "date": rebalance_date,
                    "strategy_name": strategy_name,
                    "ticker": ticker,
                    "action": action_type,
                    "reason": "kept_within_hold_band" if action_type == "HOLD" else "new_buy",
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "weight": float(new_weights.get(ticker, 0.0)),
                    "holding_days": int(holdings[ticker]["holding_days"]),
                    "rebalance_frequency_days": rebalance_frequency_days,
                    "enter_rank": enter_rank,
                    "hold_rank": hold_rank,
                    "max_holding_days": max_holding_days,
                }
            )

        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": 5,
                "top_n": top_n,
                "selected_count": selected_count,
                "qualified_count": len(ranked),
                "threshold_pass_count": len(ranked),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": float(selected["weight"].sum()) if selected_count else 0.0,
                "average_holding_days": float(np.mean([meta["holding_days"] for meta in holdings.values()])) if holdings else 0.0,
                "max_turnover_per_rebalance": max_turnover_per_rebalance,
                "rebalance_frequency_days": rebalance_frequency_days,
                "enter_rank": enter_rank,
                "hold_rank": hold_rank,
                "max_holding_days": max_holding_days,
            }
        )

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows), pd.DataFrame(action_rows)


def simulate_precomputed_panels(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float]],
    top_n: int,
    cost_bps: float,
    strategy_name: str,
    score_transform=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    portfolio_value = 10000.0
    spy_value = 10000.0

    for rebalance_date, panel, spy_return in panels:
        scored = panel.copy()
        if score_transform is not None:
            scored = score_transform(scored, rebalance_date)
        scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
        selected = scored.head(top_n).copy()
        selected_count = len(selected)
        if selected_count == 0:
            turnover = sum(abs(weight) for weight in previous_weights.values())
            transaction_cost = turnover * cost_bps / 10000.0
            net_return = -transaction_cost
            portfolio_value *= 1 + net_return
            spy_value *= 1 + spy_return
            weekly_rows.append(
                {
                    "date": rebalance_date,
                    "strategy_name": strategy_name,
                    "holding_period_days": 5,
                    "top_n": top_n,
                    "selected_count": 0,
                    "qualified_count": len(scored),
                    "threshold_pass_count": len(scored),
                    "gross_return": 0.0,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "net_return": net_return,
                    "spy_return": spy_return,
                    "excess_return": net_return - spy_return,
                    "portfolio_value": portfolio_value,
                    "spy_value": spy_value,
                    "exposure": 0.0,
                }
            )
            previous_weights = {}
            continue

        weight = 1.0 / selected_count
        selected["weight"] = weight
        selected["rank"] = np.arange(1, selected_count + 1)
        new_weights = dict(zip(selected["ticker"], selected["weight"]))
        turnover = sum(abs(new_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in set(new_weights) | set(previous_weights))
        gross_return = float((selected["weight"] * selected["future_return_used"]).sum())
        transaction_cost = turnover * cost_bps / 10000.0
        net_return = gross_return - transaction_cost
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return
        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": 5,
                "top_n": top_n,
                "selected_count": selected_count,
                "qualified_count": len(scored),
                "threshold_pass_count": len(scored),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": 1.0,
            }
        )
        holding_rows.extend(
            selected[
                [
                    column
                    for column in [
                        "ticker",
                        "sector",
                        "score",
                        "rank",
                        "weight",
                        "historical_rating_score",
                        "historical_positive_rating_ratio",
                        "historical_negative_rating_ratio",
                        "net_upgrade_score_30d",
                        "downgrade_count_30d",
                        "relative_strength_21d",
                        "relevance_weighted_sentiment_7d",
                        "sentiment_change_7d_vs_30d",
                        "negative_news_ratio_7d",
                        "volatility_21d",
                    ]
                    if column in selected.columns
                ]
            ]
            .assign(date=rebalance_date, strategy_name=strategy_name)
            .to_dict("records")
        )
        previous_weights = new_weights

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows)


def top_signal_reasons(row: pd.Series) -> str:
    contribution_columns = [column for column in row.index if column.endswith("_contribution")]
    if not contribution_columns:
        return ""
    contribution_pairs = []
    for column in contribution_columns:
        contribution_pairs.append((column.replace("_contribution", ""), float(row[column])))
    top = sorted(contribution_pairs, key=lambda item: abs(item[1]), reverse=True)[:3]
    return ", ".join(f"{name}={value:+.3f}" for name, value in top)


def suggested_rebalance_date(latest_feature_date: pd.Timestamp, holding_period_days: int) -> pd.Timestamp:
    return latest_feature_date + pd.offsets.BDay(holding_period_days)


def recommended_strategy_summary_line(recommended: RecommendedStrategyConfig) -> str:
    return (
        f"`{recommended.strategy_name}` | {recommended.holding_period_days}D | top_n={recommended.top_n} | "
        f"{recommended.position_sizing} | allow_cash={recommended.allow_cash} | threshold={recommended.threshold} | "
        f"regime_filter={recommended.regime_filter} | long_short={recommended.long_short} | cost_bps={recommended.total_cost_bps:g}"
    )


def strategy_display_from_config(recommended: RecommendedStrategyConfig) -> str:
    return strategy_display_name(recommended.strategy_name)
