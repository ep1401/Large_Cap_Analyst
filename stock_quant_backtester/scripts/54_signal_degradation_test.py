from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, summarize_backtest
from src.recommended_strategy import caveat_lines, load_runtime_and_recommended, precompute_recommended_panels, simulate_precomputed_panels
from src.utils import save_dataframe


NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30]
SEEDS = range(100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    panels = precompute_recommended_panels(features, runtime, recommended)
    rows: list[dict[str, object]] = []
    for noise_std in NOISE_LEVELS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)

            def transform(frame: pd.DataFrame, _date, local_rng=rng, local_noise=noise_std) -> pd.DataFrame:
                out = frame.copy()
                out["score"] = out["base_score"] + local_rng.normal(0.0, local_noise, len(out))
                return out

            weekly, _ = simulate_precomputed_panels(
                panels=panels,
                top_n=recommended.top_n,
                cost_bps=recommended.total_cost_bps,
                strategy_name=f"{recommended.strategy_name}_noise_{noise_std:.2f}_{seed}",
                score_transform=transform,
            )
            summary = summarize_backtest(weekly, recommended.holding_period_days, f"noise_{noise_std:.2f}_{seed}")
            summary["noise_std"] = noise_std
            summary["seed"] = seed
            summary["walk_forward_average_excess_vs_spy"] = float(
                pd.Series(
                    [
                        summary["2024_h1_excess_return_vs_spy"],
                        summary["2024_h2_excess_return_vs_spy"],
                        summary["2025_excess_return_vs_spy"],
                    ]
                ).mean()
            )
            rows.append(summary)

    results_df = pd.DataFrame(rows)
    aggregate = (
        results_df.groupby("noise_std", as_index=False)
        .agg(
            average_excess_vs_spy=("full_period_excess_return_vs_spy", "mean"),
            average_walk_forward_excess=("walk_forward_average_excess_vs_spy", "mean"),
            pct_runs_beating_spy=("full_period_excess_return_vs_spy", lambda s: float((s > 0).mean())),
            drawdown_p25=("max_drawdown", lambda s: float(s.quantile(0.25))),
            drawdown_median=("max_drawdown", "median"),
            drawdown_p75=("max_drawdown", lambda s: float(s.quantile(0.75))),
        )
    )
    save_dataframe(runtime.tables_dir / "signal_degradation_test.csv", results_df)

    fragile = bool(aggregate.loc[aggregate["noise_std"] == 0.05, "pct_runs_beating_spy"].iloc[0] < 0.70)
    report_lines = [
        "# Signal Degradation Test",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Fragile to small score changes (0.05 noise): {fragile}.",
        "",
        dataframe_to_markdown(aggregate.round(4)),
    ]
    (runtime.reports_dir / "signal_degradation_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'signal_degradation_test.csv'}")
    print(f"Saved {runtime.reports_dir / 'signal_degradation_test.md'}")


if __name__ == "__main__":
    main()
