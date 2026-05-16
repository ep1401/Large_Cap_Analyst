from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("18_validate_historical_ratings.py")), run_name="__main__")
