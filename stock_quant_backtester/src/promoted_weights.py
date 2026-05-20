from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FINAL_5D_WEIGHT_COMPONENT_ORDER = [
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "net_upgrade_score_30d",
    "downgrade_count_30d",
    "relative_strength_21d",
    "relevance_weighted_sentiment_7d",
    "sentiment_change_7d_vs_30d",
    "volatility_21d",
    "breakout_63d",
    "negative_news_flag",
    "recent_downgrade_flag",
]
FINAL_5D_GOOD_COMPONENTS = {
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "net_upgrade_score_30d",
    "relative_strength_21d",
    "relevance_weighted_sentiment_7d",
    "sentiment_change_7d_vs_30d",
    "breakout_63d",
}
FINAL_5D_BAD_COMPONENTS = set(FINAL_5D_WEIGHT_COMPONENT_ORDER) - FINAL_5D_GOOD_COMPONENTS
PROMOTED_5D_WEIGHTS_ARTIFACT = "promoted_tuned_weights_5d_no_snapshot.json"


def _project_root(project_root: Path | None = None) -> Path:
    return project_root or Path(__file__).resolve().parents[1]


def promoted_weights_artifact_path(project_root: Path | None = None) -> Path:
    return _project_root(project_root) / "configs" / PROMOTED_5D_WEIGHTS_ARTIFACT


def legacy_weight_search_path(project_root: Path | None = None) -> Path:
    return _project_root(project_root) / "outputs" / "tables" / "weight_search_5d_no_snapshot.csv"


def _normalize_final_5d_weights(weights: dict[str, float], max_abs_weight: float = 0.35) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for component in FINAL_5D_WEIGHT_COMPONENT_ORDER:
        value = float(weights.get(component, 0.0))
        if component in FINAL_5D_GOOD_COMPONENTS and value < 0:
            raise ValueError(f"{component} must have a non-negative weight.")
        if component in FINAL_5D_BAD_COMPONENTS and value > 0:
            raise ValueError(f"{component} must have a non-positive weight.")
        normalized[component] = value

    abs_sum = sum(abs(value) for value in normalized.values())
    if abs_sum <= 0:
        raise ValueError("At least one 5D component weight must be non-zero.")
    normalized = {component: value / abs_sum for component, value in normalized.items()}
    largest_abs = max(abs(value) for value in normalized.values())
    if largest_abs > max_abs_weight + 1e-9:
        raise ValueError(f"Normalized 5D weights exceed max_abs_weight={max_abs_weight}: {largest_abs:.4f}")
    return normalized


def _missing_weights_message(project_root: Path | None = None) -> str:
    artifact_path = promoted_weights_artifact_path(project_root)
    csv_path = legacy_weight_search_path(project_root)
    return (
        "Missing frozen promoted 5D tuned weights required for "
        "`final_quant_5d_weight_tuned_no_snapshot` and its low-turnover forward/paper-trading variant. "
        f"Expected committed artifact `{artifact_path}`. "
        f"Legacy fallback `{csv_path}` is optional and may exist only in local research outputs. "
        "Do not retune weights during the 2026 forward or daily paper-trading workflow; those runs must use "
        "pre-selected frozen weights chosen before the forward period."
    )


def _load_promoted_weights_from_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights_payload = payload.get("weights", payload)
    missing = [component for component in FINAL_5D_WEIGHT_COMPONENT_ORDER if component not in weights_payload]
    if missing:
        raise ValueError(f"Promoted weights artifact is missing components: {', '.join(missing)}")
    weights = {component: float(weights_payload[component]) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER}
    return _normalize_final_5d_weights(weights)


def _load_promoted_weights_from_csv(path: Path) -> dict[str, float]:
    weights_df = pd.read_csv(path)
    if "promoted" not in weights_df.columns:
        raise ValueError(f"Legacy tuned weight search results are missing the `promoted` column: {path}")
    promoted = weights_df.loc[weights_df["promoted"].fillna(False).astype(bool)].copy()
    if promoted.empty:
        raise ValueError(f"No promoted tuned 5D model is available in {path}.")
    row = promoted.iloc[0]
    weights = {component: float(row[f"weight_{component}"]) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER}
    return _normalize_final_5d_weights(weights)


def load_promoted_final_5d_tuned_weights(project_root: Path | None = None) -> dict[str, float]:
    artifact_path = promoted_weights_artifact_path(project_root)
    if artifact_path.exists():
        return _load_promoted_weights_from_json(artifact_path)

    csv_path = legacy_weight_search_path(project_root)
    if csv_path.exists():
        return _load_promoted_weights_from_csv(csv_path)

    raise FileNotFoundError(_missing_weights_message(project_root))


def assert_promoted_final_5d_tuned_weights_available(project_root: Path | None = None) -> None:
    load_promoted_final_5d_tuned_weights(project_root)
