"""Generate reproducible OULAD training artifacts from the canonical Neon source.

``oulad_kongo_full.csv`` is retained only as a temporary compatibility filename.
Its content is real OULAD data and deliberately contains no Kongo columns.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_sources import KEY_COLUMNS, TARGET_COLUMNS, load_training_mart


FULL_ARTIFACT_NAME = "oulad_kongo_full.csv"
FEATURES_ARTIFACT_NAME = "features.csv"
TARGETS_ARTIFACT_NAME = "targets.csv"
METADATA_ARTIFACT_NAME = "oulad_training_metadata.json"


def _validate_mart(frame: pd.DataFrame) -> None:
    required_columns = set(KEY_COLUMNS + TARGET_COLUMNS)
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Training mart is missing required columns: {sorted(missing_columns)}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Training mart must have one row per enrollment key")
    kongo_columns = [column for column in frame.columns if "kongo" in column.lower()]
    if kongo_columns:
        raise ValueError(f"Training mart must not contain Kongo columns: {kongo_columns}")


def write_training_artifacts(
    frame: pd.DataFrame,
    output_dir: Path,
    cutoff_day: int,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Path]:
    """Write stable CSV artifacts split by enrollment keys, features, and targets."""
    _validate_mart(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    full = frame.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    features = full.drop(columns=TARGET_COLUMNS)
    targets = full[KEY_COLUMNS + TARGET_COLUMNS]

    paths = {
        "full": output_dir / FULL_ARTIFACT_NAME,
        "features": output_dir / FEATURES_ARTIFACT_NAME,
        "targets": output_dir / TARGETS_ARTIFACT_NAME,
        "metadata": output_dir / METADATA_ARTIFACT_NAME,
    }
    full.to_csv(paths["full"], index=False, float_format="%.6f")
    features.to_csv(paths["features"], index=False, float_format="%.6f")
    targets.to_csv(paths["targets"], index=False, float_format="%.6f")

    timestamp = generated_at or datetime.now(timezone.utc)
    metadata = {
        "artifact_version": 1,
        "cutoff_day": cutoff_day,
        "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "source": "postgresql_neon_via_duckdb_read_only",
        "sql_file": "sql/oulad_training_mart.sql",
        "enrollment_key": KEY_COLUMNS,
        "row_count": len(full),
        "feature_column_count": len(features.columns) - len(KEY_COLUMNS),
        "target_column_count": len(TARGET_COLUMNS),
        "full_artifact_compatibility": {
            "filename": FULL_ARTIFACT_NAME,
            "reason": "temporary legacy filename only; content is OULAD and has no Kongo columns",
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def generate_training_artifacts(cutoff_day: int, output_dir: Path) -> dict[str, Path]:
    """Extract the enrollment mart from Neon and write the four training artifacts."""
    return write_training_artifacts(load_training_mart(cutoff_day), output_dir, cutoff_day)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate real OULAD training artifacts from Neon.")
    parser.add_argument("--cutoff-day", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data")
    args = parser.parse_args()

    paths = generate_training_artifacts(args.cutoff_day, args.output_dir)
    print(f"Generated {len(paths)} OULAD artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
