from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_features import build_feature_panel
from src.config import Config
from src.utils import str_to_bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-current-snapshot-analyst", default="true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    prices_path = config.processed_dir / "prices_all.csv"
    if not prices_path.exists():
        raise SystemExit("Missing data/processed/prices_all.csv. Run scripts/01_fetch_prices.py before building features.")
    start_date = args.start_date or config.start_date
    end_date = args.end_date or config.end_date
    window_label = f"{start_date}_{end_date}"
    sentiment_path = config.processed_dir / f"news_sentiment_daily_{window_label}.csv"
    if not sentiment_path.exists():
        sentiment_path = config.processed_dir / "news_sentiment_daily.csv"
    print(f"Feature panel window: {start_date} to {end_date}")
    features = build_feature_panel(
        prices_path=prices_path,
        universe_path=config.universe_path,
        analyst_path=config.processed_dir / "analyst_features.csv",
        sentiment_path=sentiment_path,
        market_sentiment_path=config.processed_dir / "market_sentiment_daily.csv",
        market_regime_path=config.processed_dir / "market_regime_daily.csv",
        historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
        historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
        historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
        output_path=config.final_dir / "features_panel.csv",
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=config.benchmark,
        use_current_snapshot_analyst=str_to_bool(args.use_current_snapshot_analyst),
    )
    features.to_csv(config.final_dir / f"features_panel_{window_label}.csv", index=False)
    sentiment_panel = features.loc[(features["date"] >= pd.Timestamp(start_date)) & (features["date"] < pd.Timestamp(end_date))].copy()
    sentiment_panel.to_csv(config.final_dir / f"features_panel_sentiment_{window_label}.csv", index=False)
    sentiment_rows_with_data = float(features["article_count_30d"].fillna(0).gt(0).mean()) if "article_count_30d" in features.columns else 0.0
    historical_rows_with_data = float(features["historical_rating_count_data_available"].fillna(False).mean()) if "historical_rating_count_data_available" in features.columns else 0.0
    sentiment_min_date = features.loc[features["article_count_30d"].fillna(0).gt(0), "date"].min() if "article_count_30d" in features.columns else pd.NaT
    sentiment_max_date = features.loc[features["article_count_30d"].fillna(0).gt(0), "date"].max() if "article_count_30d" in features.columns else pd.NaT
    historical_min_date = features.loc[features["historical_rating_count_data_available"].fillna(False), "historical_rating_record_date"].min() if "historical_rating_record_date" in features.columns else pd.NaT
    historical_max_date = features.loc[features["historical_rating_count_data_available"].fillna(False), "historical_rating_record_date"].max() if "historical_rating_record_date" in features.columns else pd.NaT
    print(f"Saved features rows: {len(features)}")
    print(f"Saved window-specific feature rows: {len(sentiment_panel)}")
    print(f"Feature panel min date: {features['date'].min().date() if not features.empty else 'n/a'}")
    print(f"Feature panel max date: {features['date'].max().date() if not features.empty else 'n/a'}")
    print(f"Number of tickers: {features['ticker'].nunique() if not features.empty else 0}")
    print(f"Sentiment min date: {sentiment_min_date.date() if pd.notna(sentiment_min_date) else 'n/a'}")
    print(f"Sentiment max date: {sentiment_max_date.date() if pd.notna(sentiment_max_date) else 'n/a'}")
    print(f"Historical rating count min date: {historical_min_date.date() if pd.notna(historical_min_date) else 'n/a'}")
    print(f"Historical rating count max date: {historical_max_date.date() if pd.notna(historical_max_date) else 'n/a'}")
    print(f"Percent rows with sentiment data: {sentiment_rows_with_data:.2%}")
    print(f"Percent rows with historical rating count data: {historical_rows_with_data:.2%}")

    full_window_label = config.full_analysis_window_label
    if full_window_label != window_label:
        full_features = build_feature_panel(
            prices_path=prices_path,
            universe_path=config.universe_path,
            analyst_path=config.processed_dir / "analyst_features.csv",
            sentiment_path=sentiment_path,
            market_sentiment_path=config.processed_dir / "market_sentiment_daily.csv",
            market_regime_path=config.processed_dir / "market_regime_daily.csv",
            historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
            historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
            historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
            historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
            output_path=config.final_dir / f"features_panel_{full_window_label}.csv",
            start_date=config.full_backtest_start_date,
            end_date=config.full_backtest_end_date,
            benchmark_ticker=config.benchmark,
            use_current_snapshot_analyst=str_to_bool(args.use_current_snapshot_analyst),
        )
        print(f"Saved full research window feature rows: {len(full_features)}")
        print(
            "Full research window date range: "
            f"{full_features['date'].min().date() if not full_features.empty else 'n/a'} to "
            f"{full_features['date'].max().date() if not full_features.empty else 'n/a'}"
        )


if __name__ == "__main__":
    main()
