from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

def main() -> None:
    print(
        "Alpha Vantage sentiment fetching is currently disabled. "
        "The backtest now runs with EODHD prices and optional FMP analyst data only."
    )


if __name__ == "__main__":
    main()
