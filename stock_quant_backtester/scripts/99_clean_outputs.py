from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config


def _collect_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _delete_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-only", action="store_true")
    parser.add_argument("--include-processed", action="store_true")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--include-raw-cache", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()

    targets: list[tuple[str, Path]] = [
        ("outputs/tables", config.tables_dir),
        ("outputs/reports", config.reports_dir),
        ("outputs/charts", config.charts_dir),
    ]
    if args.include_processed:
        targets.append(("data/processed", config.processed_dir))
    if args.include_final:
        targets.append(("data/final", config.final_dir))
    if args.include_raw_cache:
        targets.append(("data/raw", config.raw_dir))

    files_to_delete: list[Path] = []
    for _, root in targets:
        files_to_delete.extend(_collect_files(root))

    print("Cleanup targets:")
    for label, root in targets:
        print(f"- {label}: {root}")
    print(f"Files matched: {len(files_to_delete)}")

    if args.list:
        for path in files_to_delete:
            print(path)
        return

    if args.include_raw_cache and not args.yes:
        raise SystemExit("Refusing to delete raw API cache without --yes.")
    if (args.include_processed or args.include_final) and not args.yes:
        raise SystemExit("Pass --yes to delete processed or final datasets.")
    if not args.yes:
        raise SystemExit("Pass --yes to delete the listed files.")

    _delete_files(files_to_delete)

    for _, root in targets:
        if root.exists() and root.is_dir() and root == config.raw_dir:
            continue
        if root.exists() and not any(root.iterdir()):
            continue
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass

    print(f"Deleted files: {len(files_to_delete)}")


if __name__ == "__main__":
    main()
