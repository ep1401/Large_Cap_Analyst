from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analyst_grade_utils import classify_grade_action
from src.utils import load_dataframe, save_dataframe


HISTORICAL_GRADE_FEATURE_COLUMNS = [
    "analyst_grade_event_count_7d",
    "analyst_grade_event_count_30d",
    "analyst_grade_event_count_90d",
    "upgrade_count_7d",
    "upgrade_count_30d",
    "upgrade_count_90d",
    "downgrade_count_7d",
    "downgrade_count_30d",
    "downgrade_count_90d",
    "maintain_count_30d",
    "net_upgrade_score_7d",
    "net_upgrade_score_30d",
    "net_upgrade_score_90d",
    "avg_new_grade_score_30d",
    "avg_new_grade_score_90d",
    "avg_grade_delta_30d",
    "positive_grade_ratio_30d",
    "negative_grade_ratio_30d",
    "days_since_last_upgrade",
    "days_since_last_downgrade",
    "recent_downgrade_flag_7d",
    "recent_downgrade_flag_30d",
    "historical_grade_data_available",
]


def _empty_feature_frame(reference_df: pd.DataFrame) -> pd.DataFrame:
    out = reference_df[["date", "ticker"]].copy()
    for column in HISTORICAL_GRADE_FEATURE_COLUMNS:
        if column.startswith("recent_") or column == "historical_grade_data_available":
            out[column] = False
        elif column.startswith("avg_new_grade_score"):
            out[column] = 3.0
        elif column.startswith("days_since_"):
            out[column] = np.nan
        else:
            out[column] = 0.0
    return out


def _prepare_events(grades_df: pd.DataFrame) -> pd.DataFrame:
    if grades_df.empty:
        return grades_df.copy()

    enriched = grades_df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"]).dt.normalize()
    derived = enriched.apply(
        lambda row: classify_grade_action(row.get("previous_grade"), row.get("new_grade"), row.get("raw_action") or row.get("action")),
        axis=1,
        result_type="expand",
    )
    enriched = pd.concat([enriched, derived], axis=1)
    enriched["event_count"] = 1.0
    enriched["upgrade_count"] = enriched["is_upgrade"].astype(float)
    enriched["downgrade_count"] = enriched["is_downgrade"].astype(float)
    enriched["maintain_count"] = enriched["is_maintain"].astype(float)
    enriched["positive_grade_count"] = enriched["is_positive_grade"].astype(float)
    enriched["negative_grade_count"] = enriched["is_negative_grade"].astype(float)
    enriched["new_grade_score_value"] = pd.to_numeric(enriched["new_grade_score"], errors="coerce")
    enriched["grade_delta_value"] = pd.to_numeric(enriched["grade_delta"], errors="coerce")
    enriched["new_grade_score_sum"] = enriched["new_grade_score_value"].fillna(0.0)
    enriched["new_grade_score_count"] = enriched["new_grade_score_value"].notna().astype(float)
    enriched["grade_delta_sum"] = enriched["grade_delta_value"].fillna(0.0)
    enriched["grade_delta_count"] = enriched["grade_delta_value"].notna().astype(float)
    return enriched


def build_historical_grade_features(
    grades_input: str | Path,
    feature_reference: pd.DataFrame | str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    if isinstance(feature_reference, pd.DataFrame):
        reference_df = feature_reference[["date", "ticker"]].copy()
    else:
        reference_df = load_dataframe(feature_reference, parse_dates=["date"])[["date", "ticker"]].copy()
    reference_df["date"] = pd.to_datetime(reference_df["date"]).dt.normalize()
    reference_df = reference_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grades_path = Path(grades_input)
    if not grades_path.exists():
        features_df = _empty_feature_frame(reference_df)
        save_dataframe(output_path, features_df)
        return features_df

    grades_df = load_dataframe(grades_path, parse_dates=["date"])
    grades_df = _prepare_events(grades_df)
    if grades_df.empty:
        features_df = _empty_feature_frame(reference_df)
        save_dataframe(output_path, features_df)
        return features_df

    output_frames: list[pd.DataFrame] = []
    for ticker, ticker_ref in reference_df.groupby("ticker", sort=False):
        ticker_ref = ticker_ref.sort_values("date").reset_index(drop=True)
        ticker_events = grades_df.loc[grades_df["ticker"] == ticker].copy()

        calendar = pd.DataFrame({"date": pd.date_range(ticker_ref["date"].min(), ticker_ref["date"].max(), freq="D")})
        calendar["ticker"] = ticker

        if ticker_events.empty:
            ticker_features = _empty_feature_frame(calendar)
        else:
            daily = (
                ticker_events.groupby("date", as_index=False)[
                    [
                        "event_count",
                        "upgrade_count",
                        "downgrade_count",
                        "maintain_count",
                        "positive_grade_count",
                        "negative_grade_count",
                        "new_grade_score_sum",
                        "new_grade_score_count",
                        "grade_delta_sum",
                        "grade_delta_count",
                    ]
                ]
                .sum()
            )
            ticker_features = calendar.merge(daily, on="date", how="left").fillna(0.0)
            grouped = ticker_features.sort_values("date")

            for window in (7, 30, 90):
                grouped[f"analyst_grade_event_count_{window}d"] = grouped["event_count"].rolling(window, min_periods=1).sum()
                grouped[f"upgrade_count_{window}d"] = grouped["upgrade_count"].rolling(window, min_periods=1).sum()
                grouped[f"downgrade_count_{window}d"] = grouped["downgrade_count"].rolling(window, min_periods=1).sum()

            grouped["maintain_count_30d"] = grouped["maintain_count"].rolling(30, min_periods=1).sum()
            grouped["net_upgrade_score_7d"] = grouped["upgrade_count_7d"] - grouped["downgrade_count_7d"]
            grouped["net_upgrade_score_30d"] = grouped["upgrade_count_30d"] - grouped["downgrade_count_30d"]
            grouped["net_upgrade_score_90d"] = grouped["upgrade_count_90d"] - grouped["downgrade_count_90d"]

            score_sum_30d = grouped["new_grade_score_sum"].rolling(30, min_periods=1).sum()
            score_count_30d = grouped["new_grade_score_count"].rolling(30, min_periods=1).sum()
            score_sum_90d = grouped["new_grade_score_sum"].rolling(90, min_periods=1).sum()
            score_count_90d = grouped["new_grade_score_count"].rolling(90, min_periods=1).sum()
            delta_sum_30d = grouped["grade_delta_sum"].rolling(30, min_periods=1).sum()
            delta_count_30d = grouped["grade_delta_count"].rolling(30, min_periods=1).sum()
            positive_count_30d = grouped["positive_grade_count"].rolling(30, min_periods=1).sum()
            negative_count_30d = grouped["negative_grade_count"].rolling(30, min_periods=1).sum()

            grouped["avg_new_grade_score_30d"] = (score_sum_30d / score_count_30d.replace(0, np.nan)).fillna(3.0)
            grouped["avg_new_grade_score_90d"] = (score_sum_90d / score_count_90d.replace(0, np.nan)).fillna(3.0)
            grouped["avg_grade_delta_30d"] = (delta_sum_30d / delta_count_30d.replace(0, np.nan)).fillna(0.0)
            grouped["positive_grade_ratio_30d"] = (
                positive_count_30d / grouped["analyst_grade_event_count_30d"].replace(0, np.nan)
            ).fillna(0.0)
            grouped["negative_grade_ratio_30d"] = (
                negative_count_30d / grouped["analyst_grade_event_count_30d"].replace(0, np.nan)
            ).fillna(0.0)

            upgrade_dates = grouped["date"].where(grouped["upgrade_count"] > 0).ffill()
            downgrade_dates = grouped["date"].where(grouped["downgrade_count"] > 0).ffill()
            grouped["days_since_last_upgrade"] = (grouped["date"] - upgrade_dates).dt.days.astype(float)
            grouped["days_since_last_downgrade"] = (grouped["date"] - downgrade_dates).dt.days.astype(float)
            grouped["recent_downgrade_flag_7d"] = grouped["days_since_last_downgrade"].le(7).fillna(False)
            grouped["recent_downgrade_flag_30d"] = grouped["days_since_last_downgrade"].le(30).fillna(False)
            grouped["historical_grade_data_available"] = grouped["event_count"].cumsum().gt(0)

            ticker_features = grouped[
                [
                    "date",
                    "ticker",
                    *HISTORICAL_GRADE_FEATURE_COLUMNS,
                ]
            ].copy()

        ticker_features = ticker_ref.merge(ticker_features, on=["date", "ticker"], how="left")
        output_frames.append(ticker_features)

    features_df = pd.concat(output_frames, ignore_index=True) if output_frames else _empty_feature_frame(reference_df)
    for column in HISTORICAL_GRADE_FEATURE_COLUMNS:
        if column.startswith("recent_") or column == "historical_grade_data_available":
            features_df[column] = features_df[column].fillna(False).astype(bool)
        elif column.startswith("avg_new_grade_score"):
            features_df[column] = pd.to_numeric(features_df[column], errors="coerce").fillna(3.0)
        elif column.startswith("days_since_"):
            features_df[column] = pd.to_numeric(features_df[column], errors="coerce")
        else:
            features_df[column] = pd.to_numeric(features_df[column], errors="coerce").fillna(0.0)

    features_df = features_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    save_dataframe(output_path, features_df)
    return features_df
