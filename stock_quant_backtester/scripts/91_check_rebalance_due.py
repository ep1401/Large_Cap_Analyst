from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ml_paper_trading import load_runtime_candidate_state


def main() -> None:
    runtime, candidate, _, _, status = load_runtime_candidate_state()
    payload = {
        "strategy_name": candidate.strategy_name,
        "rebalance_due": bool(status.rebalance_due),
        "last_rebalance_date": status.last_rebalance_date.date().isoformat() if status.last_rebalance_date is not None else None,
        "next_estimated_rebalance_date": (
            status.next_estimated_rebalance_date.date().isoformat()
            if status.next_estimated_rebalance_date is not None
            else None
        ),
        "latest_trading_date": status.latest_trading_date.date().isoformat() if status.latest_trading_date is not None else None,
        "trading_days_since_rebalance": int(status.trading_days_since_rebalance),
        "rebalance_frequency_days": int(candidate.rebalance_frequency_days),
        "portfolio_state_path": str(runtime.project_root / "data" / "paper_trading" / "ml_portfolio_state.csv"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
