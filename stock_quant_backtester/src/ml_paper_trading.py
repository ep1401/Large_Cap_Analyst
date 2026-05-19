from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config
from src.ml_candidate_monitoring import ensure_market_features
from src.research_models import load_ml_research_candidate_config
from src.utils import load_dataframe, save_dataframe


ML_PAPER_TRADING_CAVEAT_LINES = [
    "This is paper trading only, not financial advice.",
    "The ML model is frozen.",
    "New forward data is not used for retraining or tuning.",
    "Back-tested performance is hypothetical unless trades were actually paper-tracked live.",
]


@dataclass(slots=True)
class RebalanceStatus:
    rebalance_due: bool
    last_rebalance_date: pd.Timestamp | None
    next_estimated_rebalance_date: pd.Timestamp | None
    latest_trading_date: pd.Timestamp | None
    trading_days_since_rebalance: int


def ml_portfolio_state_path(project_root: Path) -> Path:
    return project_root / "data" / "paper_trading" / "ml_portfolio_state.csv"


def load_ml_portfolio_state(project_root: Path) -> pd.DataFrame:
    path = ml_portfolio_state_path(project_root)
    if not path.exists():
        return pd.DataFrame()
    return load_dataframe(
        path,
        parse_dates=["as_of_date", "last_rebalance_date", "next_estimated_rebalance_date", "latest_feature_date"],
    )


def save_ml_portfolio_state(project_root: Path, state_df: pd.DataFrame) -> Path:
    path = ml_portfolio_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(path, state_df)
    return path


def load_forward_features(runtime: Config) -> pd.DataFrame:
    path = runtime.final_dir / "features_panel_2026_forward.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing forward feature panel: {path}")
    features = load_dataframe(path, parse_dates=["date"])
    if features.empty:
        raise ValueError("Forward feature panel is empty.")
    features = ensure_market_features(runtime, features)
    features["date"] = pd.to_datetime(features["date"])
    return features


def trading_dates_from_features(features: pd.DataFrame, benchmark: str) -> list[pd.Timestamp]:
    benchmark_rows = features.loc[features["ticker"] == benchmark, ["date", "adjusted_close"]].copy()
    if benchmark_rows.empty:
        benchmark_rows = features.loc[:, ["date"]].copy()
        benchmark_rows["adjusted_close"] = 1.0
    trading_dates = (
        benchmark_rows.loc[benchmark_rows["adjusted_close"].notna(), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [pd.Timestamp(value) for value in trading_dates]


def compute_rebalance_status(
    candidate,
    trading_dates: list[pd.Timestamp],
    state_df: pd.DataFrame | None = None,
) -> RebalanceStatus:
    if not trading_dates:
        return RebalanceStatus(
            rebalance_due=True,
            last_rebalance_date=None,
            next_estimated_rebalance_date=None,
            latest_trading_date=None,
            trading_days_since_rebalance=0,
        )

    latest_trading_date = pd.Timestamp(trading_dates[-1])
    if state_df is None:
        state_df = pd.DataFrame()

    if state_df.empty or "last_rebalance_date" not in state_df.columns:
        return RebalanceStatus(
            rebalance_due=True,
            last_rebalance_date=None,
            next_estimated_rebalance_date=latest_trading_date,
            latest_trading_date=latest_trading_date,
            trading_days_since_rebalance=len(trading_dates),
        )

    rebalance_dates = pd.to_datetime(state_df["last_rebalance_date"], errors="coerce").dropna()
    if rebalance_dates.empty:
        return RebalanceStatus(
            rebalance_due=True,
            last_rebalance_date=None,
            next_estimated_rebalance_date=latest_trading_date,
            latest_trading_date=latest_trading_date,
            trading_days_since_rebalance=len(trading_dates),
        )

    last_rebalance_date = pd.Timestamp(rebalance_dates.max())
    prior_or_equal = [idx for idx, date in enumerate(trading_dates) if pd.Timestamp(date) <= last_rebalance_date]
    if not prior_or_equal:
        return RebalanceStatus(
            rebalance_due=True,
            last_rebalance_date=last_rebalance_date,
            next_estimated_rebalance_date=latest_trading_date,
            latest_trading_date=latest_trading_date,
            trading_days_since_rebalance=len(trading_dates),
        )

    last_index = prior_or_equal[-1]
    trading_days_since = max(0, len(trading_dates) - last_index - 1)
    frequency = int(candidate.rebalance_frequency_days)
    due = trading_days_since >= frequency

    target_index = last_index + frequency
    if target_index < len(trading_dates):
        next_estimated = pd.Timestamp(trading_dates[target_index])
    else:
        next_estimated = pd.Timestamp(last_rebalance_date) + pd.offsets.BDay(frequency)
    if due:
        next_estimated = latest_trading_date

    return RebalanceStatus(
        rebalance_due=due,
        last_rebalance_date=last_rebalance_date,
        next_estimated_rebalance_date=next_estimated,
        latest_trading_date=latest_trading_date,
        trading_days_since_rebalance=trading_days_since,
    )


def load_runtime_candidate_state() -> tuple[Config, object, pd.DataFrame, list[pd.Timestamp], RebalanceStatus]:
    runtime = Config.from_env()
    candidate = load_ml_research_candidate_config(runtime.project_root)
    features = load_forward_features(runtime)
    features = features.loc[features["date"] >= pd.Timestamp(candidate.forward_window_start)].copy()
    state_df = load_ml_portfolio_state(runtime.project_root)
    trading_dates = trading_dates_from_features(features, runtime.benchmark)
    status = compute_rebalance_status(candidate, trading_dates, state_df)
    return runtime, candidate, state_df, trading_dates, status
