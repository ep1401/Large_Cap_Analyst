from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


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
    start_date: str
    end_date: str
    benchmark: str
    initial_capital: float
    top_n: int
    transaction_cost_bps: float
    eodhd_calls_per_minute: int
    fmp_calls_per_minute: int

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "Config":
        load_dotenv(dotenv_path=env_path)
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        final_dir = data_dir / "final"
        outputs_dir = project_root / "outputs"
        charts_dir = outputs_dir / "charts"
        reports_dir = outputs_dir / "reports"
        tables_dir = outputs_dir / "tables"

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
            start_date=os.getenv("START_DATE", "2023-01-01"),
            end_date=os.getenv("END_DATE", "2026-01-01"),
            benchmark=os.getenv("BENCHMARK", "SPY"),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000")),
            top_n=int(os.getenv("TOP_N", "10")),
            transaction_cost_bps=float(os.getenv("TRANSACTION_COST_BPS", "10")),
            eodhd_calls_per_minute=int(os.getenv("EODHD_CALLS_PER_MINUTE", "1000")),
            fmp_calls_per_minute=int(os.getenv("FMP_CALLS_PER_MINUTE", "300")),
        )
        config.ensure_directories()
        return config

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir / "prices",
            self.raw_dir / "analyst",
            self.processed_dir,
            self.final_dir,
            self.charts_dir,
            self.reports_dir,
            self.tables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
