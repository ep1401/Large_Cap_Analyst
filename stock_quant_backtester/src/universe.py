from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load the static large-cap universe definition."""
    df = pd.read_csv(path)
    required_columns = {"ticker", "company_name", "sector"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Universe file missing required columns: {sorted(missing)}")
    return df.sort_values("ticker").reset_index(drop=True)


def get_tickers(path: str | Path) -> list[str]:
    return load_universe(path)["ticker"].tolist()

