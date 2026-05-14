from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

def main() -> None:
    print(
        "Use scripts/12_fetch_alpha_vantage_news.py for cache-first Alpha Vantage news fetching. "
        "This legacy entrypoint is kept only for compatibility."
    )


if __name__ == "__main__":
    main()
