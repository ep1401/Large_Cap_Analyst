from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.scoring import SCORE_INPUT_COLUMNS, VALID_HOLDING_PERIODS, get_future_return_columns
from src.utils import load_dataframe


def main() -> None:
    config = Config.from_env()
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])

    assert not any(column.startswith("future_") for column in SCORE_INPUT_COLUMNS), "Future return columns used in scoring."
    assert VALID_HOLDING_PERIODS == {5, 21, 63}

    expected_ranges = {21: (30, 40), 63: (10, 15)}
    for holding_period_days, (low, high) in expected_ranges.items():
        rebalance_dates = select_rebalance_dates(features, holding_period_days, config.benchmark)
        assert low <= len(rebalance_dates) <= high, (
            f"Unexpected number of rebalance periods for {holding_period_days}: {len(rebalance_dates)}"
        )

    for holding_period_days in sorted(VALID_HOLDING_PERIODS):
        weekly, holdings, diagnostics = run_weekly_backtest(
            features=features,
            holding_period_days=holding_period_days,
            benchmark=config.benchmark,
            top_n=config.top_n,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            use_regime_filter=False,
            regime_exposure=0.0,
            use_analyst_filters=False,
            analyst_count_threshold=10,
            min_avg_dollar_volume=20_000_000,
            strategy_name="technical_momentum_model",
        )
        assert config.benchmark not in holdings["ticker"].unique(), "Benchmark found in holdings."
        if not holdings.empty:
            weights_by_date = holdings.groupby("date")["weight"].sum().round(8)
            exposure_by_date = weekly.set_index("date")["exposure"].round(8)
            aligned = weights_by_date.reindex(exposure_by_date.index).fillna(0.0)
            assert (aligned == exposure_by_date).all(), "Weights do not sum to exposure."
        zero_turnover_with_cost = weekly.loc[(weekly["turnover"] == 0) & (weekly["transaction_cost"] != 0)]
        assert zero_turnover_with_cost.empty, "Transaction costs should be zero when turnover is zero."
        _, future_spy_return_column, _ = get_future_return_columns(holding_period_days)
        validation_dates = select_rebalance_dates(features, holding_period_days, config.benchmark)
        benchmark_slice = features.loc[
            (features["ticker"] == config.benchmark) & (features["date"].isin(validation_dates)),
            ["date", future_spy_return_column],
        ]
        assert benchmark_slice[future_spy_return_column].notna().all(), "Missing benchmark forward returns on rebalance dates."
        if holding_period_days == 21:
            assert len(weekly) < 50, "21-day run still looks too frequent."
        if holding_period_days == 63:
            assert len(weekly) < 20, "63-day run still looks too frequent."
        assert diagnostics["selected_count"].le(config.top_n).all()

    print("Backtest validation checks passed.")


if __name__ == "__main__":
    main()
