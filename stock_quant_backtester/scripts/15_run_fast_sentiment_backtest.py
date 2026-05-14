from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = Config.from_env()
    suffix = "1y"

    _run([sys.executable, "scripts/00_cache_status.py"], cwd=project_root)
    _run([sys.executable, "scripts/12_fetch_alpha_vantage_news.py"], cwd=project_root)
    _run([sys.executable, "scripts/13_build_news_sentiment.py"], cwd=project_root)
    _run([sys.executable, "scripts/04_build_features.py"], cwd=project_root)
    _run(
        [
            sys.executable,
            "scripts/14_compare_sentiment_models.py",
            "--features-path",
            str(config.final_dir / "features_panel_sentiment_1y.csv"),
            "--start-date",
            config.sentiment_start_date,
            "--end-date",
            config.sentiment_end_date,
            "--output-suffix",
            suffix,
        ],
        cwd=project_root,
    )


if __name__ == "__main__":
    main()
