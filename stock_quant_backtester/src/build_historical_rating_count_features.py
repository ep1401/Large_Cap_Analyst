from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_dataframe, save_dataframe


HISTORICAL_RATING_COUNT_FEATURE_COLUMNS = [
    "historical_rating_count_data_available",
    "historical_rating_record_date",
    "days_since_historical_rating_update",
    "historical_total_ratings",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "historical_neutral_rating_ratio",
    "historical_buy_hold_sell_score",
    "historical_rating_score",
    "historical_rating_score_change_30d",
    "historical_rating_score_change_90d",
    "historical_positive_ratio_change_30d",
    "historical_negative_ratio_change_30d",
    "historical_negative_rating_increase_30d",
    "historical_positive_rating_increase_30d",
]

_RATING_BASE_COLUMNS = [
    "historical_total_ratings",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "historical_neutral_rating_ratio",
    "historical_buy_hold_sell_score",
    "historical_rating_score",
]


def _empty_feature_frame(reference_df: pd.DataFrame) -> pd.DataFrame:
    out = reference_df[["date", "ticker"]].copy()
    out["historical_rating_count_data_available"] = False
    out["historical_rating_record_date"] = pd.NaT
    out["days_since_historical_rating_update"] = np.nan
    out["historical_total_ratings"] = 0
    out["historical_positive_rating_ratio"] = 0.0
    out["historical_negative_rating_ratio"] = 0.0
    out["historical_neutral_rating_ratio"] = 0.0
    out["historical_buy_hold_sell_score"] = 0.0
    out["historical_rating_score"] = 3.0
    out["historical_rating_score_change_30d"] = 0.0
    out["historical_rating_score_change_90d"] = 0.0
    out["historical_positive_ratio_change_30d"] = 0.0
    out["historical_negative_ratio_change_30d"] = 0.0
    out["historical_negative_rating_increase_30d"] = False
    out["historical_positive_rating_increase_30d"] = False
    return out


def _prepare_reference(feature_reference: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(feature_reference, pd.DataFrame):
        reference_df = feature_reference[["date", "ticker"]].copy()
    else:
        reference_df = load_dataframe(feature_reference, parse_dates=["date"])[["date", "ticker"]].copy()
    reference_df["date"] = pd.to_datetime(reference_df["date"]).dt.normalize()
    return reference_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _prepare_rating_counts(rating_counts_input: str | Path) -> pd.DataFrame:
    rating_counts = load_dataframe(rating_counts_input, parse_dates=["date"])
    if rating_counts.empty:
        return rating_counts
    rating_counts = rating_counts.copy()
    rating_counts["date"] = pd.to_datetime(rating_counts["date"]).dt.normalize()
    for column in _RATING_BASE_COLUMNS:
        rating_counts[column] = pd.to_numeric(rating_counts[column], errors="coerce")
    rating_counts = (
        rating_counts.sort_values(["ticker", "date", "historical_total_ratings"], ascending=[True, True, False])
        .drop_duplicates(subset=["ticker", "date"], keep="first")
        .reset_index(drop=True)
    )
    return rating_counts


def _merge_asof_snapshot(reference_df: pd.DataFrame, rating_counts: pd.DataFrame, offset_days: int = 0) -> pd.DataFrame:
    left = reference_df[["date", "ticker"]].copy()
    if offset_days:
        left["lookup_date"] = left["date"] - pd.Timedelta(days=offset_days)
    else:
        left["lookup_date"] = left["date"]

    merged_frames: list[pd.DataFrame] = []
    for ticker, ticker_ref in left.groupby("ticker", sort=False):
        ticker_ref = ticker_ref.sort_values("lookup_date").reset_index(drop=True)
        ticker_ratings = rating_counts.loc[rating_counts["ticker"] == ticker].copy()
        if ticker_ratings.empty:
            ticker_ref = ticker_ref.assign(
                rating_date=pd.NaT,
                historical_total_ratings=np.nan,
                historical_positive_rating_ratio=np.nan,
                historical_negative_rating_ratio=np.nan,
                historical_neutral_rating_ratio=np.nan,
                historical_buy_hold_sell_score=np.nan,
                historical_rating_score=np.nan,
            )
        else:
            ticker_ratings = ticker_ratings.sort_values("date").rename(columns={"date": "rating_date"})
            ticker_ref = pd.merge_asof(
                ticker_ref,
                ticker_ratings[
                    [
                        "rating_date",
                        "historical_total_ratings",
                        "historical_positive_rating_ratio",
                        "historical_negative_rating_ratio",
                        "historical_neutral_rating_ratio",
                        "historical_buy_hold_sell_score",
                        "historical_rating_score",
                    ]
                ],
                left_on="lookup_date",
                right_on="rating_date",
                direction="backward",
            )
        merged_frames.append(ticker_ref)
    return pd.concat(merged_frames, ignore_index=True) if merged_frames else left


def build_historical_rating_count_features(
    rating_counts_input: str | Path,
    feature_reference: pd.DataFrame | str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    reference_df = _prepare_reference(feature_reference)
    rating_counts_path = Path(rating_counts_input)
    if not rating_counts_path.exists():
        features_df = _empty_feature_frame(reference_df)
        save_dataframe(output_path, features_df)
        return features_df

    rating_counts = _prepare_rating_counts(rating_counts_path)
    if rating_counts.empty:
        features_df = _empty_feature_frame(reference_df)
        save_dataframe(output_path, features_df)
        return features_df

    current = _merge_asof_snapshot(reference_df, rating_counts, offset_days=0)
    prior_30 = _merge_asof_snapshot(reference_df, rating_counts, offset_days=30)
    prior_90 = _merge_asof_snapshot(reference_df, rating_counts, offset_days=90)

    features_df = reference_df.copy()
    features_df["historical_rating_record_date"] = pd.to_datetime(current["rating_date"]).dt.normalize()
    features_df["historical_rating_count_data_available"] = features_df["historical_rating_record_date"].notna()
    features_df["days_since_historical_rating_update"] = (
        features_df["date"] - features_df["historical_rating_record_date"]
    ).dt.days.astype(float)

    for column in _RATING_BASE_COLUMNS:
        default = 3.0 if column == "historical_rating_score" else 0.0
        features_df[column] = pd.to_numeric(current[column], errors="coerce").fillna(default)

    features_df["historical_total_ratings"] = features_df["historical_total_ratings"].astype(int)
    features_df["historical_rating_score_change_30d"] = (
        features_df["historical_rating_score"] - pd.to_numeric(prior_30["historical_rating_score"], errors="coerce").fillna(3.0)
    )
    features_df["historical_rating_score_change_90d"] = (
        features_df["historical_rating_score"] - pd.to_numeric(prior_90["historical_rating_score"], errors="coerce").fillna(3.0)
    )
    features_df["historical_positive_ratio_change_30d"] = (
        features_df["historical_positive_rating_ratio"]
        - pd.to_numeric(prior_30["historical_positive_rating_ratio"], errors="coerce").fillna(0.0)
    )
    features_df["historical_negative_ratio_change_30d"] = (
        features_df["historical_negative_rating_ratio"]
        - pd.to_numeric(prior_30["historical_negative_rating_ratio"], errors="coerce").fillna(0.0)
    )
    features_df["historical_negative_rating_increase_30d"] = features_df["historical_negative_ratio_change_30d"] > 0
    features_df["historical_positive_rating_increase_30d"] = features_df["historical_positive_ratio_change_30d"] > 0

    missing_mask = ~features_df["historical_rating_count_data_available"]
    features_df.loc[missing_mask, "historical_total_ratings"] = 0
    features_df.loc[missing_mask, "historical_rating_score"] = 3.0
    for column in [
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_neutral_rating_ratio",
        "historical_buy_hold_sell_score",
        "historical_rating_score_change_30d",
        "historical_rating_score_change_90d",
        "historical_positive_ratio_change_30d",
        "historical_negative_ratio_change_30d",
    ]:
        features_df.loc[missing_mask, column] = 0.0
    features_df.loc[missing_mask, "historical_negative_rating_increase_30d"] = False
    features_df.loc[missing_mask, "historical_positive_rating_increase_30d"] = False

    features_df = features_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    save_dataframe(output_path, features_df)
    return features_df

