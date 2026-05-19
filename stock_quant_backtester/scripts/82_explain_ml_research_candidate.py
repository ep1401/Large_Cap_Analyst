from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.ml_candidate_monitoring import ensure_market_features, load_frozen_ml_context, ml_report_caveat_lines
from src.no_snapshot_research import dataframe_to_markdown
from src.research_models import ML_ALLOWED_FEATURES, ML_TARGET_COLUMN
from src.utils import load_dataframe, save_dataframe


FEATURE_GROUPS = {
    "historical_rating_counts": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
    },
    "historical_grade_events": {
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "recent_downgrade_flag_30d",
    },
    "stock_sentiment": {
        "relevance_weighted_sentiment_7d",
        "relevance_weighted_sentiment_30d",
        "sentiment_change_7d_vs_30d",
        "negative_news_ratio_7d",
    },
    "market_sentiment": {
        "market_sentiment_7d",
        "market_sentiment_30d",
        "market_negative_news_ratio_7d",
        "percent_tickers_positive_sentiment_7d",
        "percent_tickers_negative_sentiment_7d",
    },
    "market_regime": {
        "market_risk_score",
        "spy_return_21d",
        "spy_volatility_21d",
        "spy_drawdown_from_63d_high",
        "spy_above_sma_50",
        "spy_above_sma_200",
    },
    "technical_momentum": {
        "relative_strength_21d",
        "relative_strength_63d",
        "distance_to_63d_high",
        "breakout_63d",
    },
    "volatility_risk": {
        "volatility_21d",
        "beta_to_spy_63d",
    },
}


def _feature_group(name: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if name in features:
            return group
    return "other"


def main() -> None:
    runtime, candidate, artifact, _ = load_frozen_ml_context()
    features_path = runtime.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    features = ensure_market_features(runtime, features)
    features = features.loc[(features["ticker"] != runtime.benchmark) & features[ML_TARGET_COLUMN].notna()].copy()
    validation_df = features.loc[(features["date"] >= pd.Timestamp("2025-01-01")) & (features["date"] <= pd.Timestamp("2025-12-31"))].copy()

    estimator = artifact["estimator"]
    x_val = validation_df[ML_ALLOWED_FEATURES].copy()
    y_val = pd.to_numeric(validation_df[ML_TARGET_COLUMN], errors="coerce").fillna(0.0)

    model = estimator.named_steps.get("model")
    native_available = hasattr(model, "feature_importances_")
    native_importance = getattr(model, "feature_importances_", None)
    if native_available and native_importance is not None:
        importance_df = pd.DataFrame(
            {
                "feature_name": ML_ALLOWED_FEATURES,
                "importance_mean": native_importance,
                "importance_std": np.nan,
                "importance_source": "native_feature_importances",
            }
        )
    else:
        perm = permutation_importance(estimator, x_val, y_val, n_repeats=20, random_state=42, scoring="neg_mean_squared_error")
        importance_df = pd.DataFrame(
            {
                "feature_name": ML_ALLOWED_FEATURES,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
                "importance_source": "permutation_importance",
            }
        )
    importance_df["feature_group"] = importance_df["feature_name"].map(_feature_group)
    importance_df["abs_importance_mean"] = importance_df["importance_mean"].abs()
    importance_df = importance_df.sort_values("abs_importance_mean", ascending=False).reset_index(drop=True)
    save_dataframe(runtime.tables_dir / "ml_feature_importance.csv", importance_df)

    group_df = (
        importance_df.groupby("feature_group", as_index=False)
        .agg(
            total_abs_importance=("abs_importance_mean", "sum"),
            average_abs_importance=("abs_importance_mean", "mean"),
            feature_count=("feature_name", "count"),
        )
        .sort_values("total_abs_importance", ascending=False)
        .reset_index(drop=True)
    )
    save_dataframe(runtime.tables_dir / "ml_feature_group_importance.csv", group_df)

    dominant_group_share = float(group_df["total_abs_importance"].iloc[0] / group_df["total_abs_importance"].sum()) if not group_df.empty else float("nan")
    report_lines = [
        "# ML Research Candidate Explainability",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Model type: `{candidate.model_type}`",
        f"- Importance source: `{importance_df['importance_source'].iloc[0] if not importance_df.empty else 'unknown'}`",
        "",
        "## Top 20 Features",
        "",
        dataframe_to_markdown(importance_df.head(20).round(6)),
        "",
        "## Feature Groups",
        "",
        dataframe_to_markdown(group_df.round(6)),
        "",
        "## Readout",
        "",
        f"- Top feature group: `{group_df['feature_group'].iloc[0] if not group_df.empty else 'n/a'}`",
        f"- Market sentiment/regime features matter: {str(any(group_df['feature_group'].isin(['market_sentiment', 'market_regime']) & (group_df['total_abs_importance'] > 0))).lower()}",
        f"- Stock sentiment matters: {str('stock_sentiment' in group_df['feature_group'].tolist()).lower()}",
        f"- Ratings/events matter: {str(any(group_df['feature_group'].isin(['historical_rating_counts', 'historical_grade_events']))).lower()}",
        f"- Technicals dominate: {str((not group_df.empty) and group_df['feature_group'].iloc[0] in {'technical_momentum', 'volatility_risk'}).lower()}",
        f"- Model appears too dependent on one feature group: {str(pd.notna(dominant_group_share) and dominant_group_share > 0.60).lower()}",
    ]
    (runtime.reports_dir / "ml_research_candidate_explainability.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
