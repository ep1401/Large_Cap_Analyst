from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.scoring import SNAPSHOT_ANALYST_STRATEGIES, strategy_analyst_data_mode
from src.utils import load_dataframe


NO_SNAPSHOT_STRATEGIES = {
    "historical_rating_counts_model",
    "historical_rating_counts_plus_sentiment",
    "historical_rating_counts_plus_events",
    "historical_rating_counts_plus_events_sentiment",
    "final_quant_model_1y_no_snapshot",
}
SNAPSHOT_FIELDS = {
    "consensus_upside",
    "low_target_upside",
    "high_target_upside",
    "median_target",
    "last_month_target_upside",
    "last_quarter_target_upside",
    "last_year_target_upside",
    "all_time_target_upside",
    "last_month_avg_price_target",
    "last_quarter_avg_price_target",
    "last_year_avg_price_target",
    "all_time_avg_price_target",
}


def _validate_asof_logic(features: pd.DataFrame, rating_counts: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    sample_pool = features.loc[features["ticker"] != Config.from_env().benchmark].copy()
    sample_size = min(25, len(sample_pool))
    sampled = sample_pool.sample(sample_size, random_state=42) if sample_size else sample_pool

    for row in sampled.itertuples(index=False):
        history = rating_counts.loc[(rating_counts["ticker"] == row.ticker) & (rating_counts["date"] <= row.date)].copy()
        expected = history.sort_values("date").tail(1)
        if bool(row.historical_rating_count_data_available) != (not expected.empty):
            failures.append(f"{row.ticker} {row.date.date()} historical_rating_count_data_available mismatch")
            continue
        if expected.empty:
            if int(row.historical_total_ratings) != 0 or float(row.historical_rating_score) != 3.0:
                failures.append(f"{row.ticker} {row.date.date()} missing-data fill mismatch")
            continue

        expected_date = pd.to_datetime(expected["date"].iloc[0]).normalize()
        if pd.to_datetime(row.historical_rating_record_date).normalize() != expected_date:
            failures.append(f"{row.ticker} {row.date.date()} asof rating record date mismatch")
        if expected_date > row.date:
            failures.append(f"{row.ticker} {row.date.date()} future rating row used")
    return failures


def main() -> None:
    config = Config.from_env()
    rating_counts_path = config.processed_dir / "historical_analyst_rating_counts.csv"
    features_path = config.final_dir / "features_panel.csv"

    if not rating_counts_path.exists():
        raise SystemExit(
            "Missing data/processed/historical_analyst_rating_counts.csv. Run scripts/16_fetch_fmp_historical_grades.py first."
        )
    if not features_path.exists():
        raise SystemExit("Missing data/final/features_panel.csv. Run scripts/04_build_features.py first.")

    rating_counts = load_dataframe(rating_counts_path, parse_dates=["date"])
    features = load_dataframe(features_path, parse_dates=["date", "historical_rating_record_date"])

    required_feature_columns = [
        "historical_rating_count_data_available",
        "historical_rating_record_date",
        "historical_total_ratings",
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
    ]
    missing_columns = [column for column in required_feature_columns if column not in features.columns]
    if missing_columns:
        raise SystemExit(f"Missing historical rating-count feature columns: {missing_columns}")

    failures = _validate_asof_logic(features, rating_counts)
    scoring_source = (Path(__file__).resolve().parents[1] / "src" / "scoring.py").read_text(encoding="utf-8")
    final_no_snapshot_block = scoring_source.split('elif strategy_name == "final_quant_model_1y_no_snapshot":', 1)[1].split(
        'elif strategy_name == "final_quant_model_1y_no_sentiment":',
        1,
    )[0]
    leaked_snapshot_fields = sorted(field for field in SNAPSHOT_FIELDS if field in final_no_snapshot_block)
    if leaked_snapshot_fields:
        failures.append(f"final_quant_model_1y_no_snapshot references snapshot fields: {leaked_snapshot_fields}")

    for strategy_name in SNAPSHOT_ANALYST_STRATEGIES:
        if strategy_analyst_data_mode(strategy_name) != "snapshot_current":
            failures.append(f"{strategy_name} analyst_data_mode should be snapshot_current")

    expected_modes = {
        "historical_rating_counts_model": "historical_rating_counts",
        "historical_rating_counts_plus_sentiment": "historical_rating_counts_plus_sentiment",
        "historical_rating_counts_plus_events": "historical_rating_counts_plus_events",
        "historical_rating_counts_plus_events_sentiment": "historical_rating_counts_plus_events_sentiment",
        "final_quant_model_1y_no_snapshot": "historical_rating_counts_plus_events_sentiment",
    }
    for strategy_name, expected_mode in expected_modes.items():
        if strategy_analyst_data_mode(strategy_name) != expected_mode:
            failures.append(f"{strategy_name} analyst_data_mode should be {expected_mode}")

    safe_missing = features.loc[~features["historical_rating_count_data_available"].fillna(False)]
    if not safe_missing.empty:
        if not (safe_missing["historical_total_ratings"].fillna(-1) == 0).all():
            failures.append("Missing historical rating-count rows should fill historical_total_ratings with 0")
        if not (safe_missing["historical_rating_score"].fillna(-1) == 3).all():
            failures.append("Missing historical rating-count rows should fill historical_rating_score with 3")

    print("Historical ratings validation summary")
    print(f"- sampled rows: {min(25, len(features.loc[features['ticker'] != config.benchmark]))}")
    print(f"- checked no-snapshot strategies: {', '.join(sorted(NO_SNAPSHOT_STRATEGIES))}")
    print(f"- validated snapshot fields excluded from final_quant_model_1y_no_snapshot: {', '.join(sorted(SNAPSHOT_FIELDS))}")
    print(f"- failures: {len(failures)}")
    if failures:
        print("- status: FAIL")
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("- status: PASS")
    print("- no sampled row showed evidence of using a grades-historical record after the feature date")
    print("- snapshot models are labeled snapshot_current")
    print("- historical rating-count models are labeled with historical analyst data modes")
    print("- missing historical rating-count data is filled safely")


if __name__ == "__main__":
    main()
