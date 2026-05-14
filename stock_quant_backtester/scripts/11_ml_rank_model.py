from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.utils import load_dataframe, save_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven results currently use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)
FEATURE_COLUMNS = [
    "return_5d",
    "return_21d",
    "return_63d",
    "relative_strength_21d",
    "relative_strength_63d",
    "volatility_21d",
    "volatility_63d",
    "distance_to_30d_high",
    "distance_to_63d_high",
    "distance_to_252d_high",
    "breakout_30d",
    "breakout_63d",
    "above_sma_50",
    "above_sma_200",
    "rsi_14",
    "atr_14",
    "beta_to_spy_63d",
    "volume_spike_ratio",
    "consensus_upside",
    "low_target_upside",
    "analyst_count",
]
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *rows])


def _run_ranked_predictions(
    features: pd.DataFrame,
    predictions: pd.Series,
    benchmark_panel: pd.DataFrame,
    benchmark: str,
    top_n: int = 10,
) -> pd.DataFrame:
    panel = features.copy()
    panel["prediction"] = predictions.values
    decision_dates = select_rebalance_dates(benchmark_panel, holding_period_days=21, benchmark=benchmark)
    portfolio_value = 10000.0
    spy_value = 10000.0
    rows = []
    for date in decision_dates:
        day = panel.loc[(panel["date"] == date) & (panel["ticker"] != benchmark) & panel["future_21d_return"].notna()].copy()
        if day.empty:
            continue
        selected = day.sort_values("prediction", ascending=False).head(top_n)
        gross_return = float(selected["future_21d_return"].mean()) if not selected.empty else 0.0
        spy_row = benchmark_panel.loc[(benchmark_panel["date"] == date) & (benchmark_panel["ticker"] == benchmark), "future_21d_spy_return"]
        if spy_row.empty:
            continue
        spy_return = float(spy_row.iloc[0])
        portfolio_value *= 1 + gross_return
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": "ml_rank_model",
                "holding_period_days": 21,
                "selected_count": len(selected),
                "qualified_count": len(day),
                "gross_return": gross_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": gross_return,
                "spy_return": spy_return,
                "excess_return": gross_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": 1.0 if len(selected) else 0.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _extract_feature_importance(model: Pipeline) -> pd.Series | None:
    estimator = model[-1]
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if hasattr(coef, "ndim") and coef.ndim > 1:
            coef = coef[0]
        return pd.Series(coef, index=FEATURE_COLUMNS).sort_values(key=lambda s: s.abs(), ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    return None


def main() -> None:
    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    train_full = features.loc[(features["date"] >= "2023-01-01") & (features["date"] <= "2024-12-31")].copy()
    test_full = features.loc[(features["date"] >= TEST_START) & (features["date"] <= TEST_END)].copy()

    train = train_full.loc[train_full["ticker"] != config.benchmark].copy()
    test = test_full.loc[test_full["ticker"] != config.benchmark].copy()
    train = train.loc[train["future_21d_excess_return"].notna()].copy()
    test = test.loc[test["future_21d_excess_return"].notna()].copy()

    X_train = train[FEATURE_COLUMNS]
    y_reg = train["future_21d_excess_return"]
    y_clf = (train["future_21d_excess_return"] > 0).astype(int)
    X_test = test[FEATURE_COLUMNS]

    models = {
        "ridge_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "hist_gradient_boosting_regressor": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(random_state=42))]),
        "hist_gradient_boosting_classifier": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(random_state=42))]),
    }

    results = []
    importances: list[tuple[str, pd.Series]] = []
    for name, model in models.items():
        if "classifier" in name or "logistic" in name:
            model.fit(X_train, y_clf)
            if hasattr(model[-1], "predict_proba"):
                preds = model.predict_proba(X_test)[:, 1]
            else:
                preds = model.predict(X_test)
        else:
            model.fit(X_train, y_reg)
            preds = model.predict(X_test)
        returns_df = _run_ranked_predictions(
            test,
            pd.Series(preds, index=test.index),
            test_full,
            config.benchmark,
        )
        if returns_df.empty:
            continue
        metrics = calculate_performance_metrics(returns_df, holding_period_days=21)
        results.append({"model_name": name, **metrics})
        importance = _extract_feature_importance(model)
        if importance is not None:
            importances.append((name, importance.head(10)))

    results_df = pd.DataFrame(results).sort_values(by=["excess_total_return", "sharpe_ratio"], ascending=False)
    save_dataframe(config.tables_dir / "ml_model_results.csv", results_df)
    best_rule_based_excess = None
    comparison_path = config.tables_dir / "model_improvement_comparison.csv"
    if comparison_path.exists():
        comparison = load_dataframe(comparison_path)
        if "test_period_excess_return_vs_spy" in comparison.columns:
            best_rule_based_excess = float(comparison["test_period_excess_return_vs_spy"].max())

    summary_lines = []
    for model_name, importance in importances:
        summary_lines.extend(
            [
                "",
                f"## Feature Importance - {model_name}",
                "",
                _dataframe_to_markdown(importance.reset_index().rename(columns={"index": "feature", 0: "importance"}).round(6)),
            ]
        )

    beats_spy = bool((results_df["excess_total_return"] > 0).any()) if not results_df.empty else False
    beats_rule_based = (
        bool((results_df["excess_total_return"] > best_rule_based_excess).any())
        if best_rule_based_excess is not None and not results_df.empty
        else False
    )
    lines = [
        "# ML Model Summary",
        "",
        f"- {IMPORTANT_CAVEAT}",
        "- The ML experiment is optional and does not replace the rule-based models.",
        f"- Beats SPY on the 2025-01-01 to 2025-12-31 test period: {beats_spy}",
        f"- Beats the best rule-based model on test-period excess return: {beats_rule_based}",
        "",
        _dataframe_to_markdown(results_df.round(6)),
    ]
    lines.extend(summary_lines)
    report_path = config.reports_dir / "ml_model_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved ML model results to {config.tables_dir / 'ml_model_results.csv'}")
    print(f"Saved ML model summary to {report_path}")


if __name__ == "__main__":
    main()
