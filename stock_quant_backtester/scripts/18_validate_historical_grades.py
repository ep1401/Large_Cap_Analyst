from __future__ import annotations

from pathlib import Path
import random
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analyst_grade_utils import classify_grade_action
from src.config import Config
from src.utils import load_dataframe


def main() -> None:
    config = Config.from_env()
    grades_path = config.processed_dir / "historical_analyst_grades.csv"
    features_path = config.final_dir / "features_panel.csv"

    if not grades_path.exists():
        raise SystemExit("Missing data/processed/historical_analyst_grades.csv. Run scripts/16_fetch_fmp_historical_grades.py first.")
    if not features_path.exists():
        raise SystemExit("Missing data/final/features_panel.csv. Run scripts/04_build_features.py first.")

    grades = load_dataframe(grades_path, parse_dates=["date"])
    features = load_dataframe(features_path, parse_dates=["date"])
    for column in [
        "historical_grade_data_available",
        "analyst_grade_event_count_30d",
        "analyst_grade_event_count_90d",
        "upgrade_count_30d",
        "downgrade_count_30d",
    ]:
        if column not in features.columns:
            raise SystemExit(f"Missing historical grade feature column: {column}")

    derived = grades.apply(
        lambda row: classify_grade_action(row.get("previous_grade"), row.get("new_grade"), row.get("raw_action") or row.get("action")),
        axis=1,
        result_type="expand",
    )
    grades = pd.concat([grades, derived], axis=1)
    grades["date"] = pd.to_datetime(grades["date"]).dt.normalize()

    sample_pool = features.loc[features["ticker"] != config.benchmark].copy()
    sample_size = min(25, len(sample_pool))
    sampled = sample_pool.sample(sample_size, random_state=42) if sample_size else sample_pool

    failures: list[str] = []
    for row in sampled.itertuples(index=False):
        event_slice = grades.loc[(grades["ticker"] == row.ticker) & (grades["date"] <= row.date)].copy()
        last_30 = event_slice.loc[event_slice["date"] > (row.date - pd.Timedelta(days=30))]
        last_90 = event_slice.loc[event_slice["date"] > (row.date - pd.Timedelta(days=90))]

        expected_available = not event_slice.empty
        expected_event_count_30d = len(last_30)
        expected_event_count_90d = len(last_90)
        expected_upgrade_count_30d = int(last_30["is_upgrade"].fillna(False).sum())
        expected_downgrade_count_30d = int(last_30["is_downgrade"].fillna(False).sum())

        if bool(row.historical_grade_data_available) != expected_available:
            failures.append(f"{row.ticker} {row.date.date()} historical_grade_data_available mismatch")
        if int(row.analyst_grade_event_count_30d) != expected_event_count_30d:
            failures.append(f"{row.ticker} {row.date.date()} analyst_grade_event_count_30d mismatch")
        if int(row.analyst_grade_event_count_90d) != expected_event_count_90d:
            failures.append(f"{row.ticker} {row.date.date()} analyst_grade_event_count_90d mismatch")
        if int(row.upgrade_count_30d) != expected_upgrade_count_30d:
            failures.append(f"{row.ticker} {row.date.date()} upgrade_count_30d mismatch")
        if int(row.downgrade_count_30d) != expected_downgrade_count_30d:
            failures.append(f"{row.ticker} {row.date.date()} downgrade_count_30d mismatch")

    print("Historical grade validation summary")
    print(f"- sampled rows: {sample_size}")
    print(f"- failures: {len(failures)}")
    if failures:
        print("- status: FAIL")
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("- status: PASS")
    print("- no sampled row showed evidence of using grade events after the feature date")


if __name__ == "__main__":
    main()
