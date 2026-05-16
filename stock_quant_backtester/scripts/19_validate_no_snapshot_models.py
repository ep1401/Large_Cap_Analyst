from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_score_fields


def main() -> None:
    failures: list[str] = []
    checked = sorted(NO_SNAPSHOT_STRATEGIES)

    for strategy_name in checked:
        used_fields = strategy_score_fields(strategy_name)
        offending = sorted(used_fields & SNAPSHOT_FIELD_COLUMNS)
        if offending:
            failures.append(f"{strategy_name}: {', '.join(offending)}")

    lines = [
        "# No Snapshot Validation",
        "",
        f"- snapshot fields checked: {', '.join(sorted(SNAPSHOT_FIELD_COLUMNS))}",
        f"- no-snapshot strategies checked: {', '.join(checked)}",
        f"- pass/fail: {'PASS' if not failures else 'FAIL'}",
    ]
    if failures:
        lines.append(f"- offending fields: {' | '.join(failures)}")
    else:
        lines.append("- offending fields: none")

    report_path = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "no_snapshot_validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
