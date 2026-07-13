"""Versioned raw feature contract shared by OULAD training and inference."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .data_sources import KEY_COLUMNS


CONTRACT_VERSION = "oulad-academic-risk-raw-features/v2"
CATEGORICAL_FEATURES = (
    "highest_education",
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
OUTCOME_COLUMNS = (
    "final_result",
    "passed",
    "performance_tier",
    "weighted_assessment_score",
    "academic_risk",
)
SENSITIVE_COLUMNS = ("gender", "disability", "region", "imd_band")


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


def validate_raw_features(
    frame: pd.DataFrame,
    source: str,
    *,
    allow_outcomes: bool = False,
) -> pd.DataFrame:
    """Validate and normalize raw inputs without fitting or imputing values."""
    required = set(KEY_COLUMNS) | set(FEATURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    forbidden_outcomes = sorted(set(OUTCOME_COLUMNS).intersection(frame.columns))
    if forbidden_outcomes and not allow_outcomes:
        raise ValueError(f"{source} must not contain target or outcome columns: {forbidden_outcomes}")
    allowed = required | (set(OUTCOME_COLUMNS) | set(SENSITIVE_COLUMNS) if allow_outcomes else set())
    unsupported = sorted(set(frame.columns).difference(allowed))
    if unsupported:
        raise ValueError(f"{source} contains unsupported columns: {unsupported}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} has duplicate enrollment keys")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
        if features[column].isna().all():
            raise ValueError(f"{source} has no usable values for {column}")
    return features
