from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.utils import LOGGER


def cache_exists(path: Path) -> bool:
    return Path(path).exists()


def load_cached_json(path: Path) -> dict:
    path = Path(path)
    LOGGER.info("Using cached file: %s", path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_cache(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving cache: %s", path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_cached_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    LOGGER.info("Using cached file: %s", path)
    return pd.read_csv(path, parse_dates=parse_dates)


def save_csv_cache(path: Path, df: pd.DataFrame, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving cache: %s", path)
    df.to_csv(path, index=index)


def should_use_cache(path: Path, force: bool = False, cache_enabled: bool = True) -> bool:
    path = Path(path)
    use_cache = cache_enabled and not force and path.exists()
    if use_cache:
        LOGGER.info("Using cached file: %s", path)
    else:
        LOGGER.info("Fetching from API: %s", path)
    return use_cache
