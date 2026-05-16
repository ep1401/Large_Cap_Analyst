from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

from src.utils import LOGGER, load_dataframe, save_dataframe


FINBERT_MODEL_NAME = "ProsusAI/finbert"


POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "confident",
    "growth",
    "improve",
    "improved",
    "improves",
    "outperform",
    "outperformed",
    "profit",
    "profits",
    "record",
    "resilient",
    "strong",
    "surge",
    "upside",
    "upgrade",
    "upgraded",
}
NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "declines",
    "downgrade",
    "downgraded",
    "drop",
    "drops",
    "fraud",
    "investigation",
    "lawsuit",
    "loss",
    "losses",
    "miss",
    "missed",
    "misses",
    "risk",
    "risks",
    "slump",
    "weak",
    "warning",
}


@dataclass(slots=True)
class SentimentResult:
    sentiment_label: str
    sentiment_score: float
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    model_used: str


class BaseSentimentScorer:
    model_used = "unknown"

    def score_texts(self, texts: list[str]) -> list[SentimentResult]:
        raise NotImplementedError


class LexiconSentimentScorer(BaseSentimentScorer):
    model_used = "lexicon_fallback"

    def score_texts(self, texts: list[str]) -> list[SentimentResult]:
        return [self._score_one(text) for text in texts]

    def _score_one(self, text: str) -> SentimentResult:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
        negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
        raw_score = positive_hits - negative_hits
        scale = max(1.0, math.sqrt(len(tokens) + 1))
        sentiment_score = max(-1.0, min(1.0, raw_score / scale))

        if sentiment_score > 0.05:
            label = "positive"
        elif sentiment_score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        directional_strength = min(0.49, abs(sentiment_score))
        if label == "positive":
            positive_prob = 0.5 + directional_strength
            negative_prob = 0.1
        elif label == "negative":
            positive_prob = 0.1
            negative_prob = 0.5 + directional_strength
        else:
            positive_prob = 0.2
            negative_prob = 0.2
        neutral_prob = max(0.0, 1.0 - positive_prob - negative_prob)

        prob_sum = positive_prob + negative_prob + neutral_prob
        positive_prob /= prob_sum
        negative_prob /= prob_sum
        neutral_prob /= prob_sum
        sentiment_score = positive_prob - negative_prob

        return SentimentResult(
            sentiment_label=label,
            sentiment_score=sentiment_score,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            model_used=self.model_used,
        )


class FinBertSentimentScorer(BaseSentimentScorer):
    model_used = "ProsusAI/finbert"

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME, local_files_only=True)
        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    def score_texts(self, texts: list[str]) -> list[SentimentResult]:
        outputs = self._pipeline(texts, batch_size=16)
        return [self._normalize_output(scores) for scores in outputs]

    def _normalize_output(self, scores: list[dict]) -> SentimentResult:
        prob_map = {str(item["label"]).lower(): float(item["score"]) for item in scores}
        positive_prob = prob_map.get("positive", 0.0)
        negative_prob = prob_map.get("negative", 0.0)
        neutral_prob = prob_map.get("neutral", 0.0)
        sentiment_score = positive_prob - negative_prob
        if positive_prob >= negative_prob and positive_prob >= neutral_prob:
            label = "positive"
        elif negative_prob >= neutral_prob:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(
            sentiment_label=label,
            sentiment_score=sentiment_score,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            model_used=self.model_used,
        )


def get_sentiment_scorer() -> BaseSentimentScorer:
    try:
        return FinBertSentimentScorer()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "FinBERT local model was unavailable on this machine. Falling back to lexicon sentiment. Error: %s",
            exc,
        )
        return LexiconSentimentScorer()


def _prepare_text_column(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["scoring_text"] = df["text"].where(df["text"].str.strip().ne(""), df["title"])
    df["scoring_text"] = df["scoring_text"].fillna("").astype(str).str.strip()
    return df.loc[df["scoring_text"].ne("")].copy()


def _provider_result(row: pd.Series) -> SentimentResult | None:
    if "provider_sentiment_score" not in row.index or pd.isna(row.get("provider_sentiment_score")):
        return None
    label = str(row.get("provider_sentiment_label") or row.get("overall_sentiment_label") or "neutral").strip().lower()
    score = float(pd.to_numeric(row.get("provider_sentiment_score"), errors="coerce"))
    if label == "positive":
        positive_prob = min(1.0, max(0.5, 0.5 + abs(score) / 2))
        negative_prob = max(0.0, 1 - positive_prob - 0.1)
        neutral_prob = 1 - positive_prob - negative_prob
    elif label == "negative":
        negative_prob = min(1.0, max(0.5, 0.5 + abs(score) / 2))
        positive_prob = max(0.0, 1 - negative_prob - 0.1)
        neutral_prob = 1 - positive_prob - negative_prob
    else:
        positive_prob = max(0.0, 0.25 + score / 2)
        negative_prob = max(0.0, 0.25 - score / 2)
        neutral_prob = max(0.0, 1 - positive_prob - negative_prob)
        label = "neutral"
    prob_sum = positive_prob + negative_prob + neutral_prob
    positive_prob /= prob_sum
    negative_prob /= prob_sum
    neutral_prob /= prob_sum
    return SentimentResult(
        sentiment_label=label,
        sentiment_score=score,
        positive_prob=positive_prob,
        negative_prob=negative_prob,
        neutral_prob=neutral_prob,
        model_used="alpha_vantage_provider",
    )


def build_news_sentiment_outputs(
    news_input_path: str | Path,
    articles_output_path: str | Path,
    daily_output_path: str | Path,
    *,
    force: bool = False,
    rescore_with_finbert: bool = False,
    prefer_finbert: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    articles_output_path = Path(articles_output_path)
    daily_output_path = Path(daily_output_path)
    required_article_columns = {"provider_relevance_score"}
    required_daily_columns = {"relevance_weighted_sentiment_1d"}

    if articles_output_path.exists() and daily_output_path.exists() and not force:
        articles_df = load_dataframe(articles_output_path, parse_dates=["published_date", "date"])
        daily_df = load_dataframe(daily_output_path, parse_dates=["date"])
        if required_article_columns.issubset(articles_df.columns) and required_daily_columns.issubset(daily_df.columns):
            return articles_df, daily_df

    news_df = load_dataframe(news_input_path, parse_dates=["published_date", "date"])
    if news_df.empty:
        empty_articles = pd.DataFrame(
            columns=[
                "published_date",
                "date",
                "ticker",
                "title",
                "sentiment_label",
                "sentiment_score",
                "positive_prob",
                "negative_prob",
                "neutral_prob",
                "model_used",
                "provider_relevance_score",
                "url",
                "site",
            ]
        )
        empty_daily = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "article_count_1d",
                "avg_sentiment_1d",
                "median_sentiment_1d",
                "positive_news_count_1d",
                "negative_news_count_1d",
                "neutral_news_count_1d",
                "positive_news_ratio_1d",
                "negative_news_ratio_1d",
                "neutral_news_ratio_1d",
                "max_negative_prob_1d",
                "max_positive_prob_1d",
                "relevance_weighted_sentiment_1d",
            ]
        )
        save_dataframe(articles_output_path, empty_articles)
        save_dataframe(daily_output_path, empty_daily)
        return empty_articles, empty_daily

    article_rows = []
    prepared = _prepare_text_column(news_df)
    use_provider = not rescore_with_finbert and not prefer_finbert and "provider_sentiment_score" in prepared.columns
    if use_provider:
        results = [_provider_result(prepared.iloc[idx]) for idx in range(len(prepared))]
        if any(result is None for result in results):
            use_provider = False

    if not use_provider:
        scorer = get_sentiment_scorer()
        results = scorer.score_texts(prepared["scoring_text"].tolist())

    for row, result in zip(prepared.itertuples(index=False), results):
        if result is None:
            continue
        article_rows.append(
            {
                "published_date": row.published_date,
                "date": pd.Timestamp(row.date).normalize(),
                "ticker": row.ticker,
                "title": row.title,
                "sentiment_label": result.sentiment_label,
                "sentiment_score": result.sentiment_score,
                "positive_prob": result.positive_prob,
                "negative_prob": result.negative_prob,
                "neutral_prob": result.neutral_prob,
                "model_used": result.model_used,
                "provider_relevance_score": pd.to_numeric(getattr(row, "provider_relevance_score", np.nan), errors="coerce"),
                "url": row.url,
                "site": getattr(row, "site", getattr(row, "source", "")),
            }
        )

    articles_df = pd.DataFrame(article_rows).sort_values(["ticker", "published_date", "title"]).reset_index(drop=True)
    daily_df = _build_daily_aggregates(articles_df)

    save_dataframe(articles_output_path, articles_df)
    save_dataframe(daily_output_path, daily_df)
    return articles_df, daily_df


def _build_daily_aggregates(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "article_count_1d",
                "avg_sentiment_1d",
                "median_sentiment_1d",
                "positive_news_count_1d",
                "negative_news_count_1d",
                "neutral_news_count_1d",
                "positive_news_ratio_1d",
                "negative_news_ratio_1d",
                "neutral_news_ratio_1d",
                "max_negative_prob_1d",
                "max_positive_prob_1d",
                "relevance_weighted_sentiment_1d",
            ]
        )

    grouped = articles_df.groupby(["date", "ticker"], as_index=False)
    weighted_score = articles_df["sentiment_score"] * articles_df["provider_relevance_score"].fillna(1.0)
    weighted_weight = articles_df["provider_relevance_score"].fillna(1.0)
    articles_with_weights = articles_df.assign(
        weighted_sentiment_score=weighted_score,
        weighted_sentiment_weight=weighted_weight,
    )
    grouped = articles_with_weights.groupby(["date", "ticker"], as_index=False)
    daily_df = grouped.agg(
        article_count_1d=("sentiment_score", "size"),
        avg_sentiment_1d=("sentiment_score", "mean"),
        median_sentiment_1d=("sentiment_score", "median"),
        positive_news_count_1d=("sentiment_label", lambda s: int((s == "positive").sum())),
        negative_news_count_1d=("sentiment_label", lambda s: int((s == "negative").sum())),
        neutral_news_count_1d=("sentiment_label", lambda s: int((s == "neutral").sum())),
        max_negative_prob_1d=("negative_prob", "max"),
        max_positive_prob_1d=("positive_prob", "max"),
        weighted_sentiment_score_sum=("weighted_sentiment_score", "sum"),
        weighted_sentiment_weight_sum=("weighted_sentiment_weight", "sum"),
    )

    article_count = daily_df["article_count_1d"].replace(0, pd.NA)
    daily_df["positive_news_ratio_1d"] = (daily_df["positive_news_count_1d"] / article_count).fillna(0.0)
    daily_df["negative_news_ratio_1d"] = (daily_df["negative_news_count_1d"] / article_count).fillna(0.0)
    daily_df["neutral_news_ratio_1d"] = (daily_df["neutral_news_count_1d"] / article_count).fillna(0.0)
    daily_df["relevance_weighted_sentiment_1d"] = (
        daily_df["weighted_sentiment_score_sum"] / daily_df["weighted_sentiment_weight_sum"].replace(0, pd.NA)
    ).fillna(0.0)
    daily_df = daily_df.drop(columns=["weighted_sentiment_score_sum", "weighted_sentiment_weight_sum"])

    return daily_df.sort_values(["ticker", "date"]).reset_index(drop=True)
