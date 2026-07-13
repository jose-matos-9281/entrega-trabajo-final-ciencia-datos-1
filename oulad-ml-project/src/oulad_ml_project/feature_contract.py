"""Versioned raw feature contract shared by OULAD training and inference."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .data_sources import KEY_COLUMNS


CONTRACT_VERSION = "oulad-risk-raw-features/v1"
CATEGORICAL_FEATURES = (
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "disability",
    "age_band",
)
NUMERIC_FEATURES = (
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "course_duration_days",
    "total_clicks",
    "active_days",
    "vle_events",
    "vle_sites",
    "has_vle_activity",
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def contract_fingerprint() -> str:
    payload = json.dumps(
        {
            "version": CONTRACT_VERSION,
            "keys": KEY_COLUMNS,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_raw_features(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and normalize raw inputs without fitting or imputing values."""
    required = set(KEY_COLUMNS) | set(FEATURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} has duplicate enrollment keys")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
        if features[column].isna().all():
            raise ValueError(f"{source} has no usable values for {column}")
    return features
