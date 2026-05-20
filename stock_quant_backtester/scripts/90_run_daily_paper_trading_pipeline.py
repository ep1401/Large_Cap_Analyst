from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import pandas as pd

sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.ml_candidate_monitoring import load_frozen_ml_context, run_frozen_ml_forward
from src.ml_paper_trading import (
    ML_PAPER_TRADING_CAVEAT_LINES,
    compute_rebalance_status,
    load_forward_features,
    load_ml_portfolio_state,
    save_ml_portfolio_state,
    trading_dates_from_features,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import load_dataframe, save_dataframe


def _run_python_script(script_name: str, extra_args: list[str] | None = None) -> None:
    script_path = PROJECT_ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script_path), *(extra_args or [])]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _ensure_required_env(config: Config, dry_run: bool) -> None:
    required = {
        "EODHD_API_KEY": config.eodhd_api_key,
        "FMP_API_KEY": config.fmp_api_key,
        "ALPHA_VANTAGE_API_KEY": config.alpha_vantage_api_key,
    }
    missing = sorted(name for name, value in required.items() if not str(value).strip())
    if missing and not dry_run:
        raise ValueError(f"Missing required environment variables for live refresh: {', '.join(missing)}")


def _relpath(runtime: Config, path: Path) -> str:
    try:
        return str(path.relative_to(runtime.project_root))
    except ValueError:
        return str(path)


def _format_missing_paths(runtime: Config, paths: list[Path]) -> str:
    return ", ".join(_relpath(runtime, path) for path in paths)


def _bootstrap_historical_feature_panel(runtime: Config, force_refresh: bool) -> None:
    refresh_args = ["--force"] if force_refresh else []
    price_window_args = ["--start-date", runtime.full_backtest_start_date, "--end-date", runtime.full_backtest_end_date]
    sentiment_window_args = ["--start-date", runtime.full_sentiment_start_date, "--end-date", runtime.full_sentiment_end_date]

    print("Historical research feature panel missing; bootstrapping clean-checkout prerequisites from APIs.")
    _run_python_script("01_fetch_prices.py", [*price_window_args, *refresh_args])
    _run_python_script("02_fetch_analyst_data.py", refresh_args)
    _run_python_script("16_fetch_fmp_historical_grades.py", [*price_window_args, *refresh_args])
    _run_python_script("12_fetch_alpha_vantage_news.py", [*sentiment_window_args, *refresh_args])
    _run_python_script("13_build_news_sentiment.py", [*sentiment_window_args, "--force"])
    _run_python_script("68_build_market_sentiment_features.py")
    _run_python_script("69_build_market_regime_features.py")
    _run_python_script("04_build_features.py", price_window_args)


def _ensure_historical_validation_inputs(runtime: Config, dry_run: bool, force_refresh: bool) -> None:
    candidate_paths = [
        runtime.final_dir / f"features_panel_{runtime.full_analysis_window_label}.csv",
        runtime.final_dir / "features_panel.csv",
    ]
    if any(path.exists() for path in candidate_paths):
        return

    if dry_run:
        raise FileNotFoundError(
            "Missing historical feature panel required for ML leakage validation: "
            f"{_format_missing_paths(runtime, candidate_paths)}. "
            "Dry-run uses existing local datasets only. Run without --dry-run once so the pipeline can fetch prices/news/grades and build the feature panel on a clean checkout."
        )

    _bootstrap_historical_feature_panel(runtime, force_refresh=force_refresh)
    if not any(path.exists() for path in candidate_paths):
        raise FileNotFoundError(
            "Historical feature panel bootstrap completed, but the expected outputs are still missing: "
            f"{_format_missing_paths(runtime, candidate_paths)}."
        )


def _ensure_forward_inputs(runtime: Config, dry_run: bool) -> None:
    forward_path = runtime.final_dir / "features_panel_2026_forward.csv"
    if forward_path.exists():
        return
    regenerate_cmd = "python stock_quant_backtester/scripts/65_run_2026_forward_test.py"
    if dry_run:
        raise FileNotFoundError(
            "Missing forward feature panel required for dry-run paper trading: "
            f"{_relpath(runtime, forward_path)}. "
            "Dry-run skips network refresh, so regenerate it first with "
            f"`{regenerate_cmd}` (add `--force-refresh` if you need to bypass caches), "
            "or run the daily pipeline once without `--dry-run` on a runner that has the required API secrets."
        )
    raise FileNotFoundError(
        "Missing forward feature panel required for ML forward validation: "
        f"{_relpath(runtime, forward_path)}. "
        "Regenerate it before running the leakage check with "
        f"`{regenerate_cmd}` (or let this pipeline refresh it automatically before validation)."
    )


def _load_latest_recommendation_snapshot(
    runtime: Config,
    candidate,
    holdings_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    latest_feature_date: pd.Timestamp,
    rebalance_due: bool,
    prior_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdings_df.empty:
        raise ValueError("Frozen ML backtest produced no holdings for recommendations.")

    latest_signal_date = pd.Timestamp(holdings_df["date"].max())
    latest_holdings = holdings_df.loc[holdings_df["date"] == latest_signal_date].copy()
    latest_holdings["latest_feature_date"] = latest_feature_date
    latest_holdings["strategy_name"] = candidate.strategy_name
    latest_holdings["model_type"] = candidate.model_type
    latest_holdings["status"] = candidate.status
    latest_holdings["position_sizing"] = candidate.position_sizing
    latest_holdings["total_cost_bps"] = float(candidate.total_cost_bps)
    latest_holdings["rebalance_frequency_days"] = int(candidate.rebalance_frequency_days)
    latest_holdings["enter_rank"] = int(candidate.enter_rank)
    latest_holdings["hold_rank"] = int(candidate.hold_rank)
    latest_holdings["max_holding_days"] = int(candidate.max_holding_days)

    previous_holdings = set()
    if not prior_state.empty and "ticker" in prior_state.columns:
        previous_holdings = set(prior_state["ticker"].astype(str).tolist())
    current_holdings = set(latest_holdings["ticker"].astype(str).tolist())
    buys = current_holdings - previous_holdings
    holds = current_holdings & previous_holdings
    sells = sorted(previous_holdings - current_holdings)

    if rebalance_due:
        latest_holdings["action"] = latest_holdings["ticker"].map(lambda ticker: "BUY" if ticker in buys else "HOLD")
        latest_holdings["reason"] = latest_holdings["action"].map(
            {
                "BUY": "rebalance_due_new_entry",
                "HOLD": "rebalance_due_existing_holding",
            }
        )
    else:
        latest_holdings["action"] = "HOLD"
        latest_holdings["reason"] = "rebalance_not_due"

    sells_df = pd.DataFrame(
        {
            "ticker": sells,
            "action": "SELL",
            "reason": "removed_on_rebalance" if rebalance_due else "tracked_only_not_due",
        }
    )
    latest_actions = actions_df.loc[actions_df["date"] == latest_signal_date].copy() if not actions_df.empty else pd.DataFrame()
    if not latest_actions.empty:
        latest_actions = latest_actions.loc[:, [column for column in ["ticker", "action", "reason"] if column in latest_actions.columns]].copy()
        latest_actions = latest_actions.loc[latest_actions["action"].eq("SELL")].drop_duplicates(subset=["ticker"], keep="last")
        if not latest_actions.empty:
            sells_df = latest_actions
    return latest_holdings.sort_values(["rank", "ticker"]).reset_index(drop=True), sells_df.reset_index(drop=True)


def _build_state_snapshot(
    recommendations_df: pd.DataFrame,
    latest_feature_date: pd.Timestamp,
    last_rebalance_date: pd.Timestamp | None,
    next_estimated_rebalance_date: pd.Timestamp | None,
    rebalance_due: bool,
) -> pd.DataFrame:
    state_df = recommendations_df.copy()
    state_df["as_of_date"] = latest_feature_date
    state_df["last_rebalance_date"] = last_rebalance_date
    state_df["next_estimated_rebalance_date"] = next_estimated_rebalance_date
    state_df["rebalance_due"] = bool(rebalance_due)
    keep_cols = [
        "as_of_date",
        "latest_feature_date",
        "last_rebalance_date",
        "next_estimated_rebalance_date",
        "rebalance_due",
        "strategy_name",
        "model_type",
        "status",
        "ticker",
        "rank",
        "score",
        "weight",
        "action",
        "reason",
        "holding_days",
        "enter_rank",
        "hold_rank",
        "max_holding_days",
        "rebalance_frequency_days",
        "total_cost_bps",
        "position_sizing",
    ]
    return state_df[[column for column in keep_cols if column in state_df.columns]].copy()


def _write_current_recommendations(runtime: Config, candidate, recommendations_df: pd.DataFrame, sells_df: pd.DataFrame, status) -> None:
    table_path = runtime.tables_dir / "current_recommendations_ml_research_candidate.csv"
    report_path = runtime.reports_dir / "ml_current_recommendations.md"
    save_dataframe(table_path, recommendations_df)

    lines = [
        "# ML Current Recommendations",
        "",
        *[f"- {line}" for line in ML_PAPER_TRADING_CAVEAT_LINES],
        "- This is a frozen ML research candidate.",
        "- 2026 data is monitoring only and is not used for retraining or tuning.",
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Latest feature date: {pd.Timestamp(recommendations_df['latest_feature_date'].max()).date()}",
        f"- Rebalance due: {str(bool(status.rebalance_due)).lower()}",
        f"- Last rebalance date: {status.last_rebalance_date.date().isoformat() if status.last_rebalance_date is not None else 'none'}",
        (
            f"- Next estimated rebalance date: {status.next_estimated_rebalance_date.date().isoformat()}"
            if status.next_estimated_rebalance_date is not None
            else "- Next estimated rebalance date: unknown"
        ),
        "",
        "## Current Holdings",
        "",
        dataframe_to_markdown(recommendations_df.round(6)),
    ]
    if not sells_df.empty:
        lines.extend(["", "## Latest Sells", "", dataframe_to_markdown(sells_df)])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(runtime: Config, candidate, status, recommendations_df: pd.DataFrame) -> None:
    watch_df = load_dataframe(runtime.tables_dir / "ml_promotion_watch_status.csv", parse_dates=["run_timestamp", "forward_start_date", "latest_date"])
    monthly_df = load_dataframe(runtime.tables_dir / "ml_forward_monthly_returns.csv") if (runtime.tables_dir / "ml_forward_monthly_returns.csv").exists() else pd.DataFrame()
    lag_df = load_dataframe(runtime.tables_dir / "ml_conservative_lag_monitor.csv") if (runtime.tables_dir / "ml_conservative_lag_monitor.csv").exists() else pd.DataFrame()
    cost_df = load_dataframe(runtime.tables_dir / "ml_forward_cost_sensitivity.csv") if (runtime.tables_dir / "ml_forward_cost_sensitivity.csv").exists() else pd.DataFrame()

    latest_watch = watch_df.iloc[-1] if not watch_df.empty else pd.Series(dtype=object)
    latest_month = monthly_df.iloc[-1] if not monthly_df.empty else pd.Series(dtype=object)
    normal_lag = lag_df.loc[lag_df["variant_name"] == "normal_features"].iloc[-1] if not lag_df.empty and (lag_df["variant_name"] == "normal_features").any() else pd.Series(dtype=object)
    cost_20 = cost_df.loc[cost_df["total_cost_bps"] == 20].iloc[-1] if not cost_df.empty and (cost_df["total_cost_bps"] == 20).any() else pd.Series(dtype=object)

    lines = [
        "# ML Candidate Dashboard",
        "",
        *[f"- {line}" for line in ML_PAPER_TRADING_CAVEAT_LINES],
        "- This is a frozen ML research candidate.",
        "- New forward data is used for monitoring only.",
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Status: `{latest_watch.get('status', candidate.status)}`",
        f"- Rebalance due today: {str(bool(status.rebalance_due)).lower()}",
        f"- Latest monitored date: {pd.Timestamp(latest_watch.get('latest_date')).date() if pd.notna(latest_watch.get('latest_date')) else pd.Timestamp(recommendations_df['latest_feature_date'].max()).date()}",
        "",
        "## Forward Snapshot",
        "",
        f"- ML return vs SPY so far: {float(latest_watch.get('ml_excess_vs_spy', 0.0)):.2%}" if not watch_df.empty else "- ML return vs SPY so far: n/a",
        f"- ML return vs rule-based so far: {float(latest_watch.get('ml_excess_vs_rule_based', 0.0)):.2%}" if not watch_df.empty else "- ML return vs rule-based so far: n/a",
        f"- ML max drawdown: {float(latest_watch.get('ml_max_drawdown', 0.0)):.2%}" if not watch_df.empty else "- ML max drawdown: n/a",
        f"- Beats SPY in forward periods: {float(latest_watch.get('periods_beating_spy_pct', 0.0)):.2%}" if not watch_df.empty else "- Beats SPY in forward periods: n/a",
        "",
        "## Latest Month",
        "",
        f"- Latest month: `{latest_month.get('month', 'n/a')}`",
        f"- ML excess vs SPY: {float(latest_month.get('ml_excess_vs_spy', 0.0)):.2%}" if not monthly_df.empty else "- ML excess vs SPY: n/a",
        f"- ML excess vs rule-based: {float(latest_month.get('ml_excess_vs_rule_based', 0.0)):.2%}" if not monthly_df.empty else "- ML excess vs rule-based: n/a",
        "",
        "## Conservative Timing Monitor",
        "",
        f"- Normal excess vs SPY: {float(normal_lag.get('excess_vs_spy', 0.0)):.2%}" if not lag_df.empty else "- Normal excess vs SPY: n/a",
        "",
        "## Cost Readout",
        "",
        f"- Excess vs SPY at 20 bps: {float(cost_20.get('excess_vs_spy', 0.0)):.2%}" if not cost_df.empty else "- Excess vs SPY at 20 bps: n/a",
        "",
        "## Current Holdings",
        "",
        dataframe_to_markdown(recommendations_df.loc[:, [column for column in ["ticker", "rank", "score", "weight", "action", "reason"] if column in recommendations_df.columns]].round(6)),
    ]
    (runtime.reports_dir / "ml_candidate_dashboard.md").write_text("\n".join(lines), encoding="utf-8")


def _refresh_forward_data(force_refresh: bool) -> None:
    extra_args = ["--force-refresh"] if force_refresh else []
    _run_python_script("65_run_2026_forward_test.py", extra_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip network refresh and use current local forward data only.")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh upstream data when not in dry-run mode.")
    args = parser.parse_args()

    runtime = Config.from_env()
    _ensure_required_env(runtime, dry_run=args.dry_run)
    _ensure_historical_validation_inputs(runtime, dry_run=args.dry_run, force_refresh=args.force_refresh)

    if not args.dry_run:
        _refresh_forward_data(force_refresh=args.force_refresh)

    _ensure_forward_inputs(runtime, dry_run=args.dry_run)

    _run_python_script("72_validate_ml_no_leakage.py")
    _run_python_script("74_validate_ml_forward_no_leakage.py")

    _run_python_script("56_generate_paper_trading_report.py", ["--features-path", str(runtime.final_dir / "features_panel_2026_forward.csv")])
    _run_python_script("80_generate_ml_promotion_watch.py")
    _run_python_script("81_generate_ml_forward_monthly_summary.py")
    _run_python_script("84_ml_forward_cost_sensitivity.py")
    _run_python_script("85_monitor_ml_conservative_lag_variant.py")
    _run_python_script("87_plot_ml_vs_spy_validation_and_forward.py")

    if (runtime.tables_dir / "forward_2026_model_vs_spy_returns.csv").exists() and (runtime.tables_dir / "current_recommendations_2026_forward.csv").exists():
        _run_python_script("73_run_ml_2026_forward_test.py")

    _, candidate, artifact, features_forward = load_frozen_ml_context()
    prior_state = load_ml_portfolio_state(runtime.project_root)
    trading_dates = trading_dates_from_features(features_forward, runtime.benchmark)
    status = compute_rebalance_status(candidate, trading_dates, prior_state)

    ml_weekly, ml_holdings, ml_actions, _ = run_frozen_ml_forward(runtime, candidate, artifact, features_forward)
    latest_feature_date = pd.Timestamp(features_forward["date"].max())
    current_recommendations_df, sells_df = _load_latest_recommendation_snapshot(
        runtime,
        candidate,
        ml_holdings,
        ml_actions,
        latest_feature_date,
        status.rebalance_due,
        prior_state,
    )

    last_rebalance_date = latest_feature_date if status.rebalance_due else status.last_rebalance_date
    next_estimated = status.next_estimated_rebalance_date
    if status.rebalance_due:
        next_index_candidates = [idx for idx, date in enumerate(trading_dates) if pd.Timestamp(date) == latest_feature_date]
        if next_index_candidates:
            next_target_idx = next_index_candidates[-1] + int(candidate.rebalance_frequency_days)
            if next_target_idx < len(trading_dates):
                next_estimated = pd.Timestamp(trading_dates[next_target_idx])
            else:
                next_estimated = latest_feature_date + pd.offsets.BDay(int(candidate.rebalance_frequency_days))
    state_snapshot = _build_state_snapshot(
        current_recommendations_df,
        latest_feature_date,
        last_rebalance_date,
        next_estimated,
        rebalance_due=False,
    )
    state_path = save_ml_portfolio_state(runtime.project_root, state_snapshot)

    final_status = compute_rebalance_status(candidate, trading_dates, state_snapshot)
    _write_current_recommendations(runtime, candidate, current_recommendations_df, sells_df, final_status)
    _write_dashboard(runtime, candidate, final_status, current_recommendations_df)

    print(f"Dry run: {str(bool(args.dry_run)).lower()}")
    print(f"Rebalance due: {str(bool(status.rebalance_due)).lower()}")
    print(f"Saved {state_path}")
    print(f"Saved {runtime.tables_dir / 'current_recommendations_ml_research_candidate.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_current_recommendations.md'}")
    print(f"Saved {runtime.reports_dir / 'ml_candidate_dashboard.md'}")


if __name__ == "__main__":
    main()
