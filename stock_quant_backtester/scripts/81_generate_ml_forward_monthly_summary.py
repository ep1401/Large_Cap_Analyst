from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")

sys.path.append(str(PROJECT_ROOT))

from src.ml_candidate_monitoring import (
    load_frozen_ml_context,
    ml_report_caveat_lines,
    run_frozen_ml_forward,
    run_frozen_rule_forward,
    summarize_backtest_frame,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import save_dataframe


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _summarize_attribution(attribution: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if attribution.empty:
        return pd.DataFrame(), "Attribution unavailable."
    summary = (
        attribution.groupby("ticker", as_index=False)
        .agg(
            contribution_to_excess_return=("contribution_to_excess_return", "sum"),
            total_contribution=("total_contribution", "sum"),
            periods_held=("date", "count"),
        )
        .sort_values("contribution_to_excess_return", ascending=False)
        .reset_index(drop=True)
    )
    neg_total = float(summary.loc[summary["contribution_to_excess_return"] < 0, "contribution_to_excess_return"].abs().sum())
    bottom_share = (
        float(summary.sort_values("contribution_to_excess_return").head(5)["contribution_to_excess_return"].abs().sum() / neg_total)
        if neg_total > 0
        else float("nan")
    )
    narrative = "Performance looked concentrated in a few names." if pd.notna(bottom_share) and bottom_share >= 0.65 else "Performance looked broad-based rather than concentrated."
    return summary, narrative


def main() -> None:
    runtime, candidate, artifact, features_forward = load_frozen_ml_context()
    ml_weekly, _, _, ml_attr = run_frozen_ml_forward(runtime, candidate, artifact, features_forward)
    rule_weekly, _, _, rule_attr = run_frozen_rule_forward(runtime, features_forward)
    ml_weekly["month"] = pd.to_datetime(ml_weekly["period_end_date"]).dt.to_period("M").astype(str)
    rule_weekly["month"] = pd.to_datetime(rule_weekly["period_end_date"]).dt.to_period("M").astype(str)
    ml_attr["month"] = pd.to_datetime(ml_attr["period_end_date"]).dt.to_period("M").astype(str)
    rule_attr["month"] = pd.to_datetime(rule_attr["period_end_date"]).dt.to_period("M").astype(str)

    monthly_rows: list[dict[str, object]] = []
    for month in sorted(ml_weekly["month"].unique().tolist()):
        ml_month = ml_weekly.loc[ml_weekly["month"] == month].copy()
        rule_month = rule_weekly.loc[rule_weekly["month"] == month].copy()
        monthly_rows.append(
            {
                "month": month,
                "ml_return": float((1.0 + ml_month["net_return"]).prod() - 1.0),
                "spy_return": float((1.0 + ml_month["spy_return"]).prod() - 1.0),
                "rule_based_return": float((1.0 + rule_month["net_return"]).prod() - 1.0) if not rule_month.empty else float("nan"),
                "ml_excess_vs_spy": float((1.0 + ml_month["net_return"]).prod() - (1.0 + ml_month["spy_return"]).prod()),
                "ml_excess_vs_rule_based": (
                    float((1.0 + ml_month["net_return"]).prod() - (1.0 + rule_month["net_return"]).prod()) if not rule_month.empty else float("nan")
                ),
                "current_drawdown": float(ml_month["model_drawdown"].iloc[-1]),
                "worst_drawdown": float(ml_month["model_drawdown"].min()),
                "turnover": float(pd.to_numeric(ml_month["turnover"], errors="coerce").mean()),
            }
        )
    monthly_df = pd.DataFrame(monthly_rows)
    save_dataframe(runtime.tables_dir / "ml_forward_monthly_returns.csv", monthly_df)

    attr_summary, overall_narrative = _summarize_attribution(ml_attr)
    latest_month = monthly_df["month"].iloc[-1]
    latest_attr_summary, latest_narrative = _summarize_attribution(ml_attr.loc[ml_attr["month"] == latest_month].copy())

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.set_facecolor("white")
    cumulative = (1.0 + monthly_df["ml_excess_vs_spy"]).cumprod() - 1.0
    ax.plot(monthly_df["month"], cumulative, color="#7c3aed", linewidth=2.2)
    ax.axhline(0.0, color="#1f2937", linewidth=1.0, alpha=0.5)
    ax.set_title("Frozen ML Monthly Excess Trend vs SPY")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Excess Return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    _save_plot(fig, runtime.charts_dir / "ml_forward_monthly_excess_trend.png")

    latest_month_row = monthly_df.iloc[-1]
    report_lines = [
        "# ML Forward Monthly Summary",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Forward months covered: {', '.join(monthly_df['month'].tolist())}",
        f"- Latest month: {latest_month}",
        "",
        "## Monthly Returns",
        "",
        dataframe_to_markdown(monthly_df.round(6)),
        "",
        "## Readout",
        "",
        f"- Latest month ML excess vs SPY: {float(latest_month_row['ml_excess_vs_spy']):.2%}",
        f"- Latest month ML excess vs rule-based: {float(latest_month_row['ml_excess_vs_rule_based']):.2%}",
        f"- Latest month helped forward result: {str(float(latest_month_row['ml_excess_vs_spy']) > 0).lower()}",
        f"- Current drawdown: {float(ml_weekly['model_drawdown'].iloc[-1]):.2%}",
        f"- Worst drawdown so far: {float(ml_weekly['model_drawdown'].min()):.2%}",
        f"- Monthly turnover latest: {float(latest_month_row['turnover']):.4f}",
        f"- Overall attribution read: {overall_narrative}",
        f"- Latest-month attribution read: {latest_narrative}",
        f"- Top contributors overall: {', '.join(attr_summary.head(5)['ticker'].tolist()) or 'none'}",
        f"- Bottom detractors overall: {', '.join(attr_summary.sort_values('contribution_to_excess_return').head(5)['ticker'].tolist()) or 'none'}",
        f"- Top contributors latest month: {', '.join(latest_attr_summary.head(5)['ticker'].tolist()) or 'none'}",
        f"- Bottom detractors latest month: {', '.join(latest_attr_summary.sort_values('contribution_to_excess_return').head(5)['ticker'].tolist()) or 'none'}",
    ]
    (runtime.reports_dir / "ml_forward_monthly_summary.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
