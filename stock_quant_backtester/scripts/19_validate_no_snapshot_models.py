from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_long_short_backtest, run_weekly_backtest
from src.config import Config
from src.recommended_strategy import (
    load_recommended_strategy_config,
    precompute_recommended_low_turnover_panels,
    run_low_turnover_recommended_backtest,
)
from src.research_models import ML_ALLOWED_FEATURES, RESEARCH_STRATEGY_FEATURES
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_score_fields
from src.utils import load_dataframe


def main() -> None:
    failures: list[str] = []
    checked = sorted(NO_SNAPSHOT_STRATEGIES)

    for strategy_name in checked:
        used_fields = strategy_score_fields(strategy_name)
        offending = sorted(used_fields & SNAPSHOT_FIELD_COLUMNS)
        if offending:
            failures.append(f"{strategy_name}: {', '.join(offending)}")

    for strategy_name, used_fields in RESEARCH_STRATEGY_FEATURES.items():
        offending = sorted(set(used_fields) & SNAPSHOT_FIELD_COLUMNS)
        if offending:
            failures.append(f"{strategy_name}: {', '.join(offending)}")

    ml_snapshot_offenders = sorted(set(ML_ALLOWED_FEATURES) & SNAPSHOT_FIELD_COLUMNS)
    if ml_snapshot_offenders:
        failures.append(f"ml_ranker_no_snapshot features use snapshot fields: {', '.join(ml_snapshot_offenders)}")
    ml_future_offenders = sorted(feature for feature in ML_ALLOWED_FEATURES if feature.startswith("future_"))
    if ml_future_offenders:
        failures.append(f"ml_ranker_no_snapshot features use future columns: {', '.join(ml_future_offenders)}")

    config = Config.from_env()
    features_path = config.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    long_short_specs = [
        ("long_short_5d_no_snapshot_100_50", 5, 1.0, 0.5),
        ("long_short_21d_no_snapshot_100_100", 21, 1.0, 1.0),
    ]
    for strategy_name, holding_period_days, long_exposure, short_exposure in long_short_specs:
        weekly, long_holdings, short_holdings, _ = run_long_short_backtest(
            features=features,
            strategy_name=strategy_name,
            holding_period_days=holding_period_days,
            long_n=10,
            short_n=10,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            benchmark=config.benchmark,
            transaction_cost_bps=config.transaction_cost_bps,
            short_borrow_bps_annual=300,
            extra_short_slippage_bps=5,
            max_single_name_weight=0.15,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
        )
        if not long_holdings.empty and long_holdings["ticker"].eq(config.benchmark).any():
            failures.append(f"{strategy_name}: benchmark found in long book")
        if not short_holdings.empty and short_holdings["ticker"].eq(config.benchmark).any():
            failures.append(f"{strategy_name}: benchmark found in short book")
        overlap = long_holdings[["date", "ticker"]].merge(short_holdings[["date", "ticker"]], on=["date", "ticker"], how="inner")
        if not overlap.empty:
            failures.append(f"{strategy_name}: long and short books overlap")
        if not long_holdings.empty:
            long_sum = long_holdings.groupby("date")["weight"].sum()
            if not long_sum.round(8).eq(round(long_exposure, 8)).all():
                failures.append(f"{strategy_name}: long weights do not sum to long exposure")
        if not short_holdings.empty:
            short_sum = short_holdings.groupby("date")["weight"].sum()
            if not short_sum.round(8).eq(round(-short_exposure, 8)).all():
                failures.append(f"{strategy_name}: short weights do not sum to short exposure")
        if not weekly.empty:
            if not (weekly["borrow_cost"] >= 0).all():
                failures.append(f"{strategy_name}: borrow cost must be non-negative")
            if not (weekly.loc[weekly["short_exposure"] == 0, "borrow_cost"] == 0).all():
                failures.append(f"{strategy_name}: borrow cost applied without short exposure")
            if not (weekly["gross_exposure"].round(8) == (weekly["long_exposure"] + weekly["short_exposure"]).round(8)).all():
                failures.append(f"{strategy_name}: gross exposure mismatch")
            if not (weekly["net_exposure"].round(8) == (weekly["long_exposure"] - weekly["short_exposure"]).round(8)).all():
                failures.append(f"{strategy_name}: net exposure mismatch")

    selective_weekly, selective_holdings, selective_diagnostics = run_weekly_backtest(
        features=features,
        strategy_name="final_quant_5d_selective_no_snapshot",
        holding_period_days=5,
        benchmark=config.benchmark,
        top_n=10,
        transaction_cost_bps=config.transaction_cost_bps,
        use_regime_filter=False,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        max_names_per_sector=4,
        position_sizing="equal_weight",
        min_historical_rating_count=5,
        min_score_threshold=0.50,
        allow_cash=True,
        min_holdings=1,
    )
    if not selective_holdings.empty and selective_holdings["ticker"].eq(config.benchmark).any():
        failures.append("final_quant_5d_selective_no_snapshot: benchmark found in holdings")
    required_columns = {"min_score_threshold", "allow_cash", "target_top_n", "cash_weight", "percent_invested"}
    missing_required = sorted(required_columns - set(selective_weekly.columns))
    if missing_required:
        failures.append(f"final_quant_5d_selective_no_snapshot: missing output columns {missing_required}")
    if not selective_weekly.empty:
        if not selective_weekly["allow_cash"].eq(True).all():  # noqa: E712
            failures.append("final_quant_5d_selective_no_snapshot: allow_cash column should be true")
        if not selective_weekly["min_score_threshold"].fillna(-1).eq(0.5).all():
            failures.append("final_quant_5d_selective_no_snapshot: min_score_threshold should be 0.50")
        if not selective_weekly["target_top_n"].eq(10).all():
            failures.append("final_quant_5d_selective_no_snapshot: target_top_n should be 10")
        if not (selective_weekly["cash_weight"].round(8) == (1 - selective_weekly["exposure"]).round(8)).all():
            failures.append("final_quant_5d_selective_no_snapshot: cash_weight should equal 1 - exposure")
        if not (selective_weekly["percent_invested"].round(8) == selective_weekly["exposure"].round(8)).all():
            failures.append("final_quant_5d_selective_no_snapshot: percent_invested should equal exposure when regime filter is off")
        if not (selective_weekly["selected_count"] <= selective_weekly["target_top_n"]).all():
            failures.append("final_quant_5d_selective_no_snapshot: selected_count exceeds target_top_n")
    if not selective_diagnostics.empty and not (selective_diagnostics["selected_count"] <= selective_diagnostics["threshold_pass_count"]).all():
        failures.append("final_quant_5d_selective_no_snapshot: threshold should be applied before top_n selection")

    recommended = load_recommended_strategy_config(config.project_root)
    if recommended.strategy_name not in NO_SNAPSHOT_STRATEGIES:
        failures.append("recommended_strategy.yaml: strategy must be a no-snapshot strategy")
    if recommended.long_short:
        failures.append("recommended_strategy.yaml: long_short must be false")
    if recommended.regime_filter != "none":
        failures.append("recommended_strategy.yaml: regime_filter must be none")
    if recommended.threshold is not None:
        failures.append("recommended_strategy.yaml: threshold must be null/none")

    if recommended.strategy_name == "final_quant_5d_weight_tuned_low_turnover_no_snapshot":
        low_turnover_panels = precompute_recommended_low_turnover_panels(features, config, recommended)
        weekly_lt, holdings_lt, _ = run_low_turnover_recommended_backtest(
            panels=low_turnover_panels,
            top_n=recommended.top_n,
            cost_bps=float(recommended.total_cost_bps),
            enter_rank=recommended.enter_rank or recommended.top_n,
            hold_rank=recommended.hold_rank or max(recommended.top_n, 20),
            max_holding_days=recommended.max_holding_days or 20,
            rebalance_frequency_days=recommended.rebalance_frequency_days or recommended.holding_period_days,
            strategy_name=recommended.strategy_name,
            max_turnover_per_rebalance=recommended.max_turnover_per_rebalance,
        )
        if not holdings_lt.empty:
            weight_sums = holdings_lt.groupby("date")["weight"].sum().round(8)
            if not ((weight_sums == 1.0) | (weight_sums == 0.0)).all():
                failures.append("recommended_strategy.yaml low-turnover mode: holdings weights do not sum to 1.0")
            if holdings_lt["ticker"].eq(config.benchmark).any():
                failures.append("recommended_strategy.yaml low-turnover mode: benchmark found in holdings")
        if not weekly_lt.empty:
            if not weekly_lt["exposure"].between(0.0, 1.0).all():
                failures.append("recommended_strategy.yaml low-turnover mode: exposure must stay within [0, 1]")
            if recommended.max_turnover_per_rebalance is not None:
                forced_turnover_allowance = 1.0 / max(recommended.top_n, 1)
                if not (weekly_lt["turnover"] <= float(recommended.max_turnover_per_rebalance) + forced_turnover_allowance + 1e-9).all():
                    failures.append("recommended_strategy.yaml low-turnover mode: turnover cap math exceeded beyond allowance")

    current_recommendations_path = config.tables_dir / "current_recommendations_final_strategy.csv"
    if current_recommendations_path.exists():
        current_recommendations = load_dataframe(current_recommendations_path, parse_dates=["date"])
        if not current_recommendations.empty:
            if not current_recommendations["strategy_name"].eq(recommended.strategy_name).all():
                failures.append("current_recommendations_final_strategy.csv: strategy_name does not match recommended_strategy.yaml")
            if not current_recommendations["holding_period_days"].eq(recommended.holding_period_days).all():
                failures.append("current_recommendations_final_strategy.csv: holding_period_days does not match recommended_strategy.yaml")
            if not current_recommendations["position_sizing"].eq(recommended.position_sizing).all():
                failures.append("current_recommendations_final_strategy.csv: position_sizing does not match recommended_strategy.yaml")
            if not current_recommendations["allow_cash"].eq(recommended.allow_cash).all():
                failures.append("current_recommendations_final_strategy.csv: allow_cash does not match recommended_strategy.yaml")
            expected_threshold = -9999 if recommended.threshold is None else float(recommended.threshold)
            actual_threshold = current_recommendations["min_score_threshold"].fillna(-9999)
            if not actual_threshold.eq(expected_threshold).all():
                failures.append("current_recommendations_final_strategy.csv: min_score_threshold does not match recommended_strategy.yaml")
            if not current_recommendations["total_cost_bps"].eq(float(recommended.total_cost_bps)).all():
                failures.append("current_recommendations_final_strategy.csv: total_cost_bps does not match recommended_strategy.yaml")
            if len(current_recommendations) > recommended.top_n:
                failures.append("current_recommendations_final_strategy.csv: row count exceeds recommended top_n")
            weight_total = current_recommendations["weight"].sum()
            if not (abs(weight_total - 1.0) < 1e-8 or abs(weight_total) < 1e-8):
                failures.append("current_recommendations_final_strategy.csv: weights do not sum to 1.0")

    lines = [
        "# No Snapshot Validation",
        "",
        f"- snapshot fields checked: {', '.join(sorted(SNAPSHOT_FIELD_COLUMNS))}",
        f"- no-snapshot strategies checked: {', '.join(checked)}",
        f"- research candidates checked: {', '.join(sorted(RESEARCH_STRATEGY_FEATURES))}",
        f"- long/short structural checks run on: {', '.join(name for name, _, _, _ in long_short_specs)}",
        "- selective cash/threshold checks run on: final_quant_5d_selective_no_snapshot",
        f"- recommended strategy config checked: {recommended.strategy_name}",
        f"- pass/fail: {'PASS' if not failures else 'FAIL'}",
    ]
    if failures:
        lines.append(f"- offending fields: {' | '.join(failures)}")
    else:
        lines.append("- offending fields: none")

    report_path = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "no_snapshot_validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
