from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.no_snapshot_research import (
    FINAL_5D_BAD_COMPONENTS,
    FINAL_5D_COMPONENT_EXPORT_COLUMNS,
    FINAL_5D_GOOD_COMPONENTS,
    FINAL_5D_WEIGHT_COMPONENT_ORDER,
    WALK_FORWARD_WINDOWS,
    build_eligible_universe,
    build_final_5d_component_frame,
    build_weight_tuned_final_quant_5d_definition,
    dataframe_to_markdown,
    fmt_pct,
    get_baseline_final_5d_weights,
    get_best_5d_config,
    load_features,
    normalize_final_5d_weights,
    run_custom_weekly_backtest,
    safe_metrics,
    slice_period,
    summarize_backtest,
)
from src.scoring import strategy_display_name
from src.utils import save_dataframe


BASELINE_STRATEGY = "final_quant_5d_no_snapshot_no_sma_filter"
TUNED_STRATEGY = "final_quant_5d_weight_tuned_no_snapshot"
BASELINE_TOP_N = 10
BASELINE_COST_BPS = 10.0
MAX_ABS_WEIGHT = 0.35
WEIGHT_SEARCH_SEED = 7
DEFAULT_RANDOM_CANDIDATES = 60


def _component_rows(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    rebalance_dates = select_rebalance_dates(features, holding_period_days=5, benchmark=config.benchmark)
    rows: list[pd.DataFrame] = []
    baseline_weights = get_baseline_final_5d_weights()
    for rebalance_date in rebalance_dates:
        day_slice = features.loc[(features["date"] == rebalance_date) & (features["ticker"] != config.benchmark)].copy()
        qualified, diagnostics = build_eligible_universe(
            day_slice=day_slice,
            holding_period_days=5,
            benchmark=config.benchmark,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            require_historical_rating_count=True,
            min_historical_rating_count=5,
            require_historical_grade_data=True,
            exclude_strong_negative_news=False,
            exclude_recent_downgrades=False,
        )
        if qualified.empty:
            continue
        component_frame = build_final_5d_component_frame(qualified, baseline_weights)
        component_frame["baseline_passes_negative_news_filter"] = ~qualified["strong_negative_news_flag"].fillna(False).astype(bool)
        component_frame["baseline_passes_recent_downgrade_filter"] = ~qualified["recent_downgrade_flag_30d"].fillna(False).astype(bool)
        component_frame["baseline_passes_all_filters"] = (
            component_frame["baseline_passes_negative_news_filter"] & component_frame["baseline_passes_recent_downgrade_filter"]
        )
        component_frame["qualified_count_before_hard_filters"] = int(diagnostics["final_pass_count"])
        keep_columns = [
            "date",
            "ticker",
            "sector",
            "score",
            "baseline_passes_negative_news_filter",
            "baseline_passes_recent_downgrade_filter",
            "baseline_passes_all_filters",
            "qualified_count_before_hard_filters",
            *FINAL_5D_COMPONENT_EXPORT_COLUMNS.values(),
        ]
        rows.append(component_frame[[column for column in keep_columns if column in component_frame.columns]].copy())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _signed_weights_from_magnitudes(magnitudes: np.ndarray) -> dict[str, float]:
    raw = {}
    for idx, component in enumerate(FINAL_5D_WEIGHT_COMPONENT_ORDER):
        sign = 1.0 if component in FINAL_5D_GOOD_COMPONENTS else -1.0
        raw[component] = float(magnitudes[idx] * sign)
    return normalize_final_5d_weights(raw, max_abs_weight=MAX_ABS_WEIGHT)


def _weights_from_row(row: pd.Series) -> dict[str, float]:
    return normalize_final_5d_weights(
        {component: float(row[f"weight_{component}"]) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER},
        max_abs_weight=MAX_ABS_WEIGHT,
    )


def _load_incumbent_weights(config: Config) -> dict[str, float] | None:
    path = config.tables_dir / "weight_search_5d_no_snapshot.csv"
    if not path.exists():
        return None
    existing = pd.read_csv(path)
    if "promoted" not in existing.columns:
        return None
    promoted = existing.loc[existing["promoted"].fillna(False).astype(bool)].copy()
    if promoted.empty:
        return None
    return _weights_from_row(promoted.iloc[0])


def _generate_candidate_weights(random_candidates: int, incumbent: dict[str, float] | None = None) -> list[dict[str, float]]:
    rng = np.random.default_rng(WEIGHT_SEARCH_SEED)
    baseline = get_baseline_final_5d_weights()
    baseline_abs = np.array([abs(baseline[component]) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER], dtype=float)
    baseline_abs = baseline_abs / baseline_abs.sum()

    candidates: list[dict[str, float]] = [baseline]
    seen = {tuple(round(candidate[component], 6) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER) for candidate in candidates}

    seed_profiles = [baseline_abs]
    if incumbent is not None:
        candidates.append(incumbent)
        seen.add(tuple(round(incumbent[component], 6) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER))
        incumbent_abs = np.array([abs(incumbent[component]) for component in FINAL_5D_WEIGHT_COMPONENT_ORDER], dtype=float)
        incumbent_abs = incumbent_abs / incumbent_abs.sum()
        seed_profiles.append(incumbent_abs)

    for seed_abs in seed_profiles:
        for component in FINAL_5D_WEIGHT_COMPONENT_ORDER:
            for delta in (0.04, 0.08):
                bumped = seed_abs.copy()
                component_idx = FINAL_5D_WEIGHT_COMPONENT_ORDER.index(component)
                bumped[component_idx] += delta
                bumped = bumped / bumped.sum()
                if bumped.max() <= MAX_ABS_WEIGHT + 1e-9:
                    normalized = _signed_weights_from_magnitudes(bumped)
                    key = tuple(round(normalized[name], 6) for name in FINAL_5D_WEIGHT_COMPONENT_ORDER)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(normalized)

    alpha_profiles = [
        1.0 + baseline_abs * 50.0,
        0.6 + baseline_abs * 25.0,
        np.full(len(FINAL_5D_WEIGHT_COMPONENT_ORDER), 1.0),
    ]
    if len(seed_profiles) > 1:
        incumbent_abs = seed_profiles[1]
        alpha_profiles.extend(
            [
                1.0 + incumbent_abs * 50.0,
                0.6 + incumbent_abs * 25.0,
            ]
        )
    while len(candidates) < random_candidates + 1:
        alpha = alpha_profiles[(len(candidates) - 1) % len(alpha_profiles)]
        sample = rng.dirichlet(alpha)
        if float(sample.max()) > MAX_ABS_WEIGHT + 1e-9:
            continue
        normalized = _signed_weights_from_magnitudes(sample)
        key = tuple(round(normalized[name], 6) for name in FINAL_5D_WEIGHT_COMPONENT_ORDER)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)

    return candidates


def _window_columns() -> list[str]:
    return [f"{label.lower().replace(' ', '_')}_excess_return_vs_spy" for label, _, _ in WALK_FORWARD_WINDOWS]


def _evaluate_candidate(
    features: pd.DataFrame,
    config: Config,
    weights: dict[str, float],
    candidate_id: str,
    max_names_per_sector: int | None,
) -> dict[str, object]:
    definition = build_weight_tuned_final_quant_5d_definition(weights)
    weekly, holdings, _ = run_custom_weekly_backtest(
        features=features,
        definition=definition,
        holding_period_days=5,
        benchmark=config.benchmark,
        top_n=BASELINE_TOP_N,
        transaction_cost_bps=BASELINE_COST_BPS,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        max_names_per_sector=max_names_per_sector,
        position_sizing="equal_weight",
    )
    summary = summarize_backtest(weekly, 5, candidate_id)
    window_columns = _window_columns()
    summary["candidate_id"] = candidate_id
    summary["strategy_name"] = TUNED_STRATEGY
    summary["display_name"] = strategy_display_name(TUNED_STRATEGY)
    summary["walk_forward_average_excess_vs_spy"] = float(pd.Series([summary[column] for column in window_columns]).mean())
    summary["worst_window_excess_vs_spy"] = float(pd.Series([summary[column] for column in window_columns]).min())
    summary["average_exposure"] = float(weekly["exposure"].mean()) if not weekly.empty else 0.0
    summary["average_selected_count"] = float(weekly["selected_count"].mean()) if not weekly.empty else 0.0
    summary["top_n"] = BASELINE_TOP_N
    summary["total_cost_bps"] = BASELINE_COST_BPS
    summary["allow_cash"] = False
    summary["min_score_threshold"] = np.nan
    summary["train_2023_excess_return_vs_spy"] = safe_metrics(slice_period(weekly, "2023-01-01", "2023-12-31"), 5)["excess_total_return"]
    summary["train_2023_to_2024_h1_excess_return_vs_spy"] = safe_metrics(
        slice_period(weekly, "2023-01-01", "2024-06-30"), 5
    )["excess_total_return"]
    summary["train_2023_to_2024_excess_return_vs_spy"] = safe_metrics(
        slice_period(weekly, "2023-01-01", "2024-12-31"), 5
    )["excess_total_return"]
    summary["selected_names_latest"] = ", ".join(holdings.loc[holdings["date"] == holdings["date"].max(), "ticker"].head(10).tolist()) if not holdings.empty else ""
    for component in FINAL_5D_WEIGHT_COMPONENT_ORDER:
        summary[f"weight_{component}"] = weights[component]
    return summary


def _baseline_summary(features: pd.DataFrame, config: Config, max_names_per_sector: int | None) -> dict[str, object]:
    weekly, _, _ = run_weekly_backtest(
        features=features,
        holding_period_days=5,
        benchmark=config.benchmark,
        top_n=BASELINE_TOP_N,
        initial_capital=config.initial_capital,
        transaction_cost_bps=BASELINE_COST_BPS,
        use_regime_filter=False,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        strategy_name=BASELINE_STRATEGY,
        max_names_per_sector=max_names_per_sector,
        position_sizing="equal_weight",
        min_historical_rating_count=5,
        allow_cash=False,
    )
    summary = summarize_backtest(weekly, 5, BASELINE_STRATEGY)
    summary["strategy_name"] = BASELINE_STRATEGY
    summary["display_name"] = strategy_display_name(BASELINE_STRATEGY)
    summary["walk_forward_average_excess_vs_spy"] = float(pd.Series([summary[column] for column in _window_columns()]).mean())
    summary["worst_window_excess_vs_spy"] = float(pd.Series([summary[column] for column in _window_columns()]).min())
    summary["average_exposure"] = float(weekly["exposure"].mean()) if not weekly.empty else 0.0
    summary["top_n"] = BASELINE_TOP_N
    summary["total_cost_bps"] = BASELINE_COST_BPS
    summary["allow_cash"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--random-candidates", type=int, default=DEFAULT_RANDOM_CANDIDATES)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)
    max_names_per_sector = best_config["max_names_per_sector"]

    component_df = _component_rows(features, config)
    save_dataframe(config.tables_dir / "final_5d_score_components.csv", component_df)

    baseline_summary = _baseline_summary(features, config, max_names_per_sector)
    incumbent_weights = _load_incumbent_weights(config)
    candidates = _generate_candidate_weights(args.random_candidates, incumbent=incumbent_weights)

    rows = []
    for idx, weights in enumerate(candidates):
        candidate_id = "baseline_normalized" if idx == 0 else f"candidate_{idx:04d}"
        rows.append(_evaluate_candidate(features, config, weights, candidate_id, max_names_per_sector))

    results_df = pd.DataFrame(rows)
    results_df["beats_spy_in_at_least_2_windows"] = results_df["windows_beating_spy"] >= 2
    results_df["turnover_vs_baseline_ratio"] = results_df["average_turnover"] / float(baseline_summary["average_turnover"])
    results_df["drawdown_delta_vs_baseline"] = results_df["max_drawdown"] - float(baseline_summary["max_drawdown"])
    results_df["walk_forward_delta_vs_baseline"] = (
        results_df["walk_forward_average_excess_vs_spy"] - float(baseline_summary["walk_forward_average_excess_vs_spy"])
    )
    results_df["worst_window_guardrail_pass"] = results_df["worst_window_excess_vs_spy"] >= max(
        -0.05, float(baseline_summary["worst_window_excess_vs_spy"]) - 0.03
    )
    results_df["turnover_guardrail_pass"] = results_df["turnover_vs_baseline_ratio"] <= 1.25
    results_df["selection_eligible"] = (
        results_df["beats_spy_in_at_least_2_windows"]
        & results_df["worst_window_guardrail_pass"]
        & results_df["turnover_guardrail_pass"]
    )

    incumbent_summary = None
    if incumbent_weights is not None:
        incumbent_row = results_df.loc[
            results_df.apply(
                lambda row: all(
                    abs(float(row[f"weight_{component}"]) - incumbent_weights[component]) <= 1e-6
                    for component in FINAL_5D_WEIGHT_COMPONENT_ORDER
                ),
                axis=1,
            )
        ]
        if not incumbent_row.empty:
            incumbent_summary = incumbent_row.iloc[0]
    if incumbent_summary is None:
        incumbent_summary = baseline_summary

    window_columns = _window_columns()
    for window_column in window_columns:
        results_df[f"beats_incumbent_{window_column}"] = (
            results_df[window_column] > float(incumbent_summary[window_column]) + 1e-9
        )
    incumbent_beat_columns = [f"beats_incumbent_{window_column}" for window_column in window_columns]
    results_df["beats_incumbent_in_all_windows"] = results_df[incumbent_beat_columns].all(axis=1)
    results_df["beats_incumbent_window_count"] = results_df[incumbent_beat_columns].sum(axis=1)
    results_df["walk_forward_delta_vs_incumbent"] = (
        results_df["walk_forward_average_excess_vs_spy"] - float(incumbent_summary["walk_forward_average_excess_vs_spy"])
    )
    results_df["drawdown_delta_vs_incumbent"] = results_df["max_drawdown"] - float(incumbent_summary["max_drawdown"])

    ranked = results_df.sort_values(
        [
            "beats_incumbent_in_all_windows",
            "selection_eligible",
            "walk_forward_average_excess_vs_spy",
            "worst_window_excess_vs_spy",
            "max_drawdown",
            "average_turnover",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    ranked["selected_candidate"] = False
    ranked["promoted"] = False
    ranked.loc[0, "selected_candidate"] = True

    promotion_pass = bool(
        ranked.loc[0, "selection_eligible"]
        and ranked.loc[0, "walk_forward_delta_vs_incumbent"] > 0
        and ranked.loc[0, "beats_incumbent_in_all_windows"]
        and ranked.loc[0, "drawdown_delta_vs_incumbent"] >= -0.03
    )
    if promotion_pass:
        ranked.loc[0, "promoted"] = True
    elif incumbent_weights is not None:
        incumbent_matches = ranked.apply(
            lambda row: all(
                abs(float(row[f"weight_{component}"]) - incumbent_weights[component]) <= 1e-6
                for component in FINAL_5D_WEIGHT_COMPONENT_ORDER
            ),
            axis=1,
        )
        if incumbent_matches.any():
            ranked.loc[incumbent_matches, "promoted"] = True

    save_dataframe(config.tables_dir / "weight_search_5d_no_snapshot.csv", ranked)

    selected = ranked.iloc[0]
    report_table = ranked[
        [
            "candidate_id",
            "walk_forward_average_excess_vs_spy",
            "2024_h1_excess_return_vs_spy",
            "2024_h2_excess_return_vs_spy",
            "2025_excess_return_vs_spy",
            "worst_window_excess_vs_spy",
            "windows_beating_spy",
            "max_drawdown",
            "average_turnover",
            "walk_forward_delta_vs_baseline",
            "walk_forward_delta_vs_incumbent",
            "beats_incumbent_in_all_windows",
        ]
    ].head(10).copy()

    weight_lines = [
        f"- `weight_{component}` = {selected[f'weight_{component}']:.4f}"
        for component in FINAL_5D_WEIGHT_COMPONENT_ORDER
    ]
    baseline_weights = get_baseline_final_5d_weights()
    baseline_weight_lines = [
        f"- `weight_{component}` = {baseline_weights[component]:.4f}"
        for component in FINAL_5D_WEIGHT_COMPONENT_ORDER
    ]

    report_lines = [
        "# Weight Search 5D No Snapshot",
        "",
        "## Baseline",
        f"- Baseline strategy: `{BASELINE_STRATEGY}`.",
        f"- Walk-forward average excess vs SPY: {fmt_pct(float(baseline_summary['walk_forward_average_excess_vs_spy']))}.",
        f"- Worst window excess vs SPY: {fmt_pct(float(baseline_summary['worst_window_excess_vs_spy']))}.",
        f"- Max drawdown: {fmt_pct(float(baseline_summary['max_drawdown']))}.",
        f"- Average turnover: {baseline_summary['average_turnover']:.4f}.",
        "",
        "## Selection Rules",
        "- Candidate must beat SPY in at least 2 of 3 walk-forward windows.",
        "- Candidate ranking prioritizes walk-forward average excess, then worst-window excess, then drawdown, then turnover.",
        "- Promotion requires a strictly better walk-forward average than the current incumbent tuned model.",
        "- Promotion also requires beating the incumbent in every walk-forward test window and no worse than -3.00 percentage points of drawdown delta vs incumbent.",
        "",
        "## Incumbent",
        f"- Incumbent reference strategy: `{'current tuned model' if incumbent_weights is not None else 'baseline fallback'}`.",
        f"- Incumbent walk-forward average excess vs SPY: {fmt_pct(float(incumbent_summary['walk_forward_average_excess_vs_spy']))}.",
        f"- Incumbent 2024 H1 excess vs SPY: {fmt_pct(float(incumbent_summary['2024_h1_excess_return_vs_spy']))}.",
        f"- Incumbent 2024 H2 excess vs SPY: {fmt_pct(float(incumbent_summary['2024_h2_excess_return_vs_spy']))}.",
        f"- Incumbent 2025 excess vs SPY: {fmt_pct(float(incumbent_summary['2025_excess_return_vs_spy']))}.",
        "",
        "## Selected Candidate",
        f"- Candidate id: `{selected['candidate_id']}`.",
        f"- Promoted to `{TUNED_STRATEGY}`: {promotion_pass}.",
        f"- Walk-forward average excess vs SPY: {fmt_pct(float(selected['walk_forward_average_excess_vs_spy']))}.",
        f"- Walk-forward delta vs baseline: {fmt_pct(float(selected['walk_forward_delta_vs_baseline']))}.",
        f"- Walk-forward delta vs incumbent: {fmt_pct(float(selected['walk_forward_delta_vs_incumbent']))}.",
        f"- Beats incumbent in all walk-forward windows: {bool(selected['beats_incumbent_in_all_windows'])}.",
        f"- Worst window excess vs SPY: {fmt_pct(float(selected['worst_window_excess_vs_spy']))}.",
        f"- Max drawdown: {fmt_pct(float(selected['max_drawdown']))}.",
        f"- Drawdown delta vs incumbent: {fmt_pct(float(selected['drawdown_delta_vs_incumbent']))}.",
        f"- Average turnover: {selected['average_turnover']:.4f}.",
        f"- Turnover vs baseline ratio: {selected['turnover_vs_baseline_ratio']:.2f}x.",
        "",
        "### Selected Weights",
        *weight_lines,
        "",
        "### Baseline Normalized Weights",
        *baseline_weight_lines,
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(report_table.round(4)),
    ]
    (config.reports_dir / "weight_search_5d_no_snapshot.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {config.tables_dir / 'final_5d_score_components.csv'}")
    print(f"Saved {config.tables_dir / 'weight_search_5d_no_snapshot.csv'}")
    print(f"Saved {config.reports_dir / 'weight_search_5d_no_snapshot.md'}")


if __name__ == "__main__":
    main()
