from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
import pandas as pd


@dataclass(slots=True)
class Config:
    """Central runtime configuration loaded from environment variables."""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    final_dir: Path
    outputs_dir: Path
    charts_dir: Path
    reports_dir: Path
    tables_dir: Path
    universe_path: Path
    eodhd_api_key: str
    fmp_api_key: str
    alpha_vantage_api_key: str
    start_date: str
    end_date: str
    sentiment_start_date: str
    sentiment_end_date: str
    sentiment_lookback_years: int
    full_backtest_start_date: str
    full_backtest_end_date: str
    full_sentiment_start_date: str
    full_sentiment_end_date: str
    full_run_force_refresh: bool
    full_run_clear_cache: bool
    full_run_clear_outputs: bool
    historical_analyst_lookback_days: int
    cache_enabled: bool
    force_refresh: bool
    news_provider: str
    benchmark: str
    initial_capital: float
    top_n: int
    transaction_cost_bps: float
    min_avg_dollar_volume: float
    analyst_count_threshold: int
    eodhd_calls_per_minute: int
    fmp_calls_per_minute: int
    alpha_vantage_requests_per_minute: int

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "Config":
        load_dotenv(dotenv_path=env_path)
        def env_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "t", "yes", "y"}

        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        final_dir = data_dir / "final"
        outputs_dir = project_root / "outputs"
        charts_dir = outputs_dir / "charts"
        reports_dir = outputs_dir / "reports"
        tables_dir = outputs_dir / "tables"
        end_date = os.getenv("END_DATE", "2026-01-01")
        sentiment_start_override = os.getenv("SENTIMENT_START_DATE")
        sentiment_end_override = os.getenv("SENTIMENT_END_DATE")
        sentiment_lookback_years = int(os.getenv("SENTIMENT_LOOKBACK_YEARS", "1"))
        full_backtest_start_date = os.getenv("FULL_BACKTEST_START_DATE", "2023-01-01")
        full_backtest_end_date = os.getenv("FULL_BACKTEST_END_DATE", "2026-01-01")
        full_sentiment_start_date = os.getenv("FULL_SENTIMENT_START_DATE", full_backtest_start_date)
        full_sentiment_end_date = os.getenv("FULL_SENTIMENT_END_DATE", full_backtest_end_date)

        if sentiment_start_override and sentiment_end_override:
            sentiment_start_date = sentiment_start_override
            sentiment_end_date = sentiment_end_override
        else:
            sentiment_end_ts = pd.Timestamp(end_date)
            sentiment_start_ts = sentiment_end_ts - pd.DateOffset(years=sentiment_lookback_years)
            sentiment_start_date = sentiment_start_ts.strftime("%Y-%m-%d")
            sentiment_end_date = sentiment_end_ts.strftime("%Y-%m-%d")

        config = cls(
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            final_dir=final_dir,
            outputs_dir=outputs_dir,
            charts_dir=charts_dir,
            reports_dir=reports_dir,
            tables_dir=tables_dir,
            universe_path=data_dir / "universe" / "large_cap_universe.csv",
            eodhd_api_key=os.getenv("EODHD_API_KEY", ""),
            fmp_api_key=os.getenv("FMP_API_KEY", ""),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            start_date=os.getenv("START_DATE", "2025-01-01"),
            end_date=end_date,
            sentiment_start_date=sentiment_start_date,
            sentiment_end_date=sentiment_end_date,
            sentiment_lookback_years=sentiment_lookback_years,
            full_backtest_start_date=full_backtest_start_date,
            full_backtest_end_date=full_backtest_end_date,
            full_sentiment_start_date=full_sentiment_start_date,
            full_sentiment_end_date=full_sentiment_end_date,
            full_run_force_refresh=env_bool("FULL_RUN_FORCE_REFRESH", False),
            full_run_clear_cache=env_bool("FULL_RUN_CLEAR_CACHE", False),
            full_run_clear_outputs=env_bool("FULL_RUN_CLEAR_OUTPUTS", True),
            historical_analyst_lookback_days=int(os.getenv("HISTORICAL_ANALYST_LOOKBACK_DAYS", "365")),
            cache_enabled=env_bool("CACHE_ENABLED", True),
            force_refresh=env_bool("FORCE_REFRESH", False),
            news_provider=os.getenv("NEWS_PROVIDER", "alpha_vantage"),
            benchmark=os.getenv("BENCHMARK", "SPY"),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000")),
            top_n=int(os.getenv("TOP_N", "10")),
            transaction_cost_bps=float(os.getenv("TRANSACTION_COST_BPS", "10")),
            min_avg_dollar_volume=float(os.getenv("MIN_AVG_DOLLAR_VOLUME", "20000000")),
            analyst_count_threshold=int(os.getenv("ANALYST_COUNT_THRESHOLD", "10")),
            eodhd_calls_per_minute=int(os.getenv("EODHD_CALLS_PER_MINUTE", "1000")),
            fmp_calls_per_minute=int(os.getenv("FMP_CALLS_PER_MINUTE", "300")),
            alpha_vantage_requests_per_minute=int(os.getenv("ALPHA_VANTAGE_REQUESTS_PER_MINUTE", "60")),
        )
        config.ensure_directories()
        return config

    @property
    def sentiment_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.sentiment_start_date)

    @property
    def sentiment_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.sentiment_end_date)

    @property
    def sentiment_window_label(self) -> str:
        return f"{self.sentiment_start_date}_{self.sentiment_end_date}"

    @property
    def analysis_window_label(self) -> str:
        return f"{self.start_date}_{self.end_date}"

    @property
    def full_analysis_window_label(self) -> str:
        return f"{self.full_backtest_start_date}_{self.full_backtest_end_date}"

    def describe_analysis_windows(self) -> str:
        return (
            f"Analysis window: {self.start_date} to {self.end_date} | "
            f"Sentiment window: {self.sentiment_start_date} to {self.sentiment_end_date}"
        )

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir / "prices",
            self.raw_dir / "prices" / "eodhd",
            self.raw_dir / "analyst",
            self.raw_dir / "analyst" / "fmp",
            self.raw_dir / "analyst" / "fmp_historical_grades",
            self.raw_dir / "news",
            self.raw_dir / "news" / "alpha_vantage",
            self.processed_dir,
            self.final_dir,
            self.charts_dir,
            self.reports_dir,
            self.tables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
