from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_features import build_feature_panel
from src.config import Config
from src.utils import str_to_bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-current-snapshot-analyst", default="true")
    args = parser.parse_args()

    config = Config.from_env()
    features = build_feature_panel(
        prices_path=config.processed_dir / "prices_all.csv",
        universe_path=config.universe_path,
        analyst_path=config.processed_dir / "analyst_features.csv",
        sentiment_path=config.processed_dir / "news_sentiment_daily.csv",
        output_path=config.final_dir / "features_panel.csv",
        benchmark_ticker=config.benchmark,
        use_current_snapshot_analyst=str_to_bool(args.use_current_snapshot_analyst),
    )
    print(f"Saved features rows: {len(features)}")


if __name__ == "__main__":
    main()
