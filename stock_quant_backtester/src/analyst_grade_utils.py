from __future__ import annotations

import re


GRADE_SCORE_MAP = {
    5.0: {"strong buy", "top pick", "conviction buy"},
    4.0: {
        "buy",
        "outperform",
        "overweight",
        "positive",
        "sector outperform",
        "market outperform",
        "accumulate",
    },
    3.0: {
        "hold",
        "neutral",
        "market perform",
        "sector perform",
        "equal weight",
        "equal-weight",
        "inline",
        "in-line",
        "peer perform",
    },
    2.0: {"sell", "underperform", "underweight", "negative", "reduce"},
    1.0: {"strong sell"},
}


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_grade_to_score(grade: str) -> float | None:
    cleaned = _clean_text(grade)
    if not cleaned:
        return None

    for score, aliases in GRADE_SCORE_MAP.items():
        if cleaned in aliases:
            return score

    if "strong buy" in cleaned:
        return 5.0
    if "strong sell" in cleaned:
        return 1.0
    if any(token in cleaned for token in ["conviction buy", "top pick"]):
        return 5.0
    if any(token in cleaned for token in ["outperform", "overweight", "positive", "accumulate", "buy"]):
        return 4.0
    if any(token in cleaned for token in ["neutral", "hold", "market perform", "sector perform", "equal weight", "peer perform", "inline"]):
        return 3.0
    if any(token in cleaned for token in ["underperform", "underweight", "negative", "reduce", "sell"]):
        return 2.0
    return None


def classify_grade_action(previous_grade, new_grade, action) -> dict:
    previous_grade_score = normalize_grade_to_score(previous_grade) if previous_grade is not None else None
    new_grade_score = normalize_grade_to_score(new_grade) if new_grade is not None else None
    cleaned_action = _clean_text(action)

    grade_delta = None
    if previous_grade_score is not None and new_grade_score is not None:
        grade_delta = new_grade_score - previous_grade_score

    is_upgrade = "upgrade" in cleaned_action
    is_downgrade = "downgrade" in cleaned_action
    is_maintain = any(token in cleaned_action for token in {"maintain", "reiterate", "initiated", "initiate"})

    if grade_delta is not None:
        if grade_delta > 0:
            is_upgrade = True
            is_downgrade = False
            is_maintain = False
        elif grade_delta < 0:
            is_upgrade = False
            is_downgrade = True
            is_maintain = False
        elif not (is_upgrade or is_downgrade):
            is_maintain = True

    if not (is_upgrade or is_downgrade or is_maintain):
        if previous_grade_score is None or new_grade_score is None:
            is_maintain = cleaned_action == ""

    return {
        "previous_grade_score": previous_grade_score,
        "new_grade_score": new_grade_score,
        "grade_delta": grade_delta,
        "is_upgrade": bool(is_upgrade),
        "is_downgrade": bool(is_downgrade),
        "is_maintain": bool(is_maintain and not is_upgrade and not is_downgrade),
        "is_positive_grade": bool(new_grade_score is not None and new_grade_score >= 4.0),
        "is_negative_grade": bool(new_grade_score is not None and new_grade_score <= 2.0),
    }
