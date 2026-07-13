"""Leakage-safe champion selection and artifact reporting for OULAD risk models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_sources import KEY_COLUMNS
from .feature_contract import (
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    contract_fingerprint,
    validate_raw_features,
)


ARTIFACT_VERSION = 2
MODEL_FILENAME = "passed_model.joblib"
MANIFEST_FILENAME = "inference_manifest.json"
TRAINING_REPORT_FILENAME = "training_report.json"
METRICS_FILENAME = "training_metrics.csv"
EVALUATION_PREDICTIONS_FILENAME = "evaluation_predictions.csv"


@dataclass(frozen=True)
class TrainingArtifacts:
    model: Path
    manifest: Path
    report: Path
    metrics: Path
    evaluation_predictions: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "model": self.model,
            "manifest": self.manifest,
            "report": self.report,
            "metrics": self.metrics,
            "evaluation_predictions": self.evaluation_predictions,
        }


def _pipeline(classifier: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                list(NUMERIC_FEATURES),
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _candidates() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": _pipeline(LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForestClassifier": _pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
        "GradientBoostingClassifier": _pipeline(GradientBoostingClassifier(n_estimators=100, random_state=42)),
    }


def _metrics(y_true: pd.Series, predicted: pd.Series, probability_passed: pd.Series) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "risk_recall": recall_score(y_true, predicted, pos_label=0, zero_division=0),
        "risk_precision": precision_score(y_true, predicted, pos_label=0, zero_division=0),
        "risk_f1": f1_score(y_true, predicted, pos_label=0, zero_division=0),
        "accuracy": accuracy_score(y_true, predicted),
        "precision_macro": precision_score(y_true, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, predicted, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, predicted, average="macro", zero_division=0),
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_true, probability_passed) if y_true.nunique() == 2 else None
    )
    return metrics


def _configuration(model: Pipeline) -> dict[str, object]:
    classifier = model.named_steps["classifier"]
    return {
        "classifier": classifier.__class__.__name__,
        "parameters": classifier.get_params(),
        "preprocessor": "categorical most-frequent imputation + one-hot; numeric median imputation + standard scaling",
    }


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    return {key: (float(values.mean()) if values.notna().any() else None)
            for key in rows[0] if (values := pd.Series([row[key] for row in rows])) is not None}


def train_risk_champion(frame: pd.DataFrame, artifacts_dir: Path, cutoff_day: int) -> TrainingArtifacts:
    """Select by grouped CV within train, then evaluate the champion once on holdout."""
    if isinstance(cutoff_day, bool) or not isinstance(cutoff_day, int):
        raise TypeError("cutoff_day must be an integer")
    if "passed" not in frame.columns or frame["passed"].isna().any() or frame["passed"].nunique() != 2:
        raise ValueError("training data must contain both non-null passed classes")

    features = validate_raw_features(frame, "training data")
    target = pd.to_numeric(frame["passed"], errors="raise").astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("passed must contain exactly the binary values 0 and 1")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_positions, test_positions = next(splitter.split(features, target, frame["id_student"]))
    X_train, X_holdout = features.iloc[train_positions], features.iloc[test_positions]
    y_train, y_holdout = target.iloc[train_positions], target.iloc[test_positions]
    if y_train.nunique() != 2 or y_holdout.nunique() != 2:
        raise ValueError("grouped train and holdout splits must each contain both passed classes")

    candidate_rows: list[dict[str, object]] = []
    train_groups = frame.iloc[train_positions]["id_student"]
    folds = min(5, train_groups.nunique())
    if folds < 2:
        raise ValueError("grouped training split needs at least two students for CV selection")
    cv = GroupKFold(n_splits=folds)
    for name, candidate in _candidates().items():
        fold_metrics = []
        for fit_positions, validation_positions in cv.split(X_train, y_train, train_groups):
            model = _pipeline(clone(candidate.named_steps["classifier"]))
            model.fit(X_train.iloc[fit_positions], y_train.iloc[fit_positions])
            predicted = pd.Series(model.predict(X_train.iloc[validation_positions]), dtype=int)
            probability = pd.Series(model.predict_proba(X_train.iloc[validation_positions])[:, 1])
            fold_metrics.append(_metrics(y_train.iloc[validation_positions].reset_index(drop=True), predicted, probability))
        candidate_rows.append({"model": name, **_mean_metrics(fold_metrics)})

    # Risk recall is the primary operational objective; remaining metrics only break ties.
    champion_row = max(
        candidate_rows,
        key=lambda row: tuple(float(row[key] or 0) for key in ("risk_recall", "risk_f1", "risk_precision", "roc_auc")),
    )
    champion_name = str(champion_row["model"])
    champion = _candidates()[champion_name].fit(X_train, y_train)
    predicted = pd.Series(champion.predict(X_holdout), index=X_holdout.index, dtype=int)
    probability_passed = pd.Series(champion.predict_proba(X_holdout)[:, 1], index=X_holdout.index)
    holdout_metrics = _metrics(y_holdout, predicted, probability_passed)
    evaluation_rows = pd.DataFrame({
        **{column: frame.iloc[test_positions][column].to_numpy() for column in KEY_COLUMNS},
        "model": champion_name, "passed_actual": y_holdout.to_numpy(),
        "prediction_passed": predicted.to_numpy(), "probability_passed": probability_passed.to_numpy(),
        "probability_risk": 1 - probability_passed.to_numpy(),
    })

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    version = uuid4().hex
    model_path = artifacts_dir / f"{version}-{MODEL_FILENAME}"
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    report_path = artifacts_dir / f"{version}-{TRAINING_REPORT_FILENAME}"
    metrics_path = artifacts_dir / f"{version}-{METRICS_FILENAME}"
    predictions_path = artifacts_dir / f"{version}-{EVALUATION_PREDICTIONS_FILENAME}"
    joblib.dump(champion, model_path)
    pd.DataFrame(candidate_rows).to_csv(metrics_path, index=False)
    evaluation_rows.to_csv(predictions_path, index=False)

    split = {
        "strategy": "GroupShuffleSplit",
        "group_column": "id_student",
        "random_state": 42,
        "test_size": 0.25,
        "train_rows": len(X_train),
        "test_rows": len(X_holdout),
        "train_students": int(frame.iloc[train_positions]["id_student"].nunique()),
        "test_students": int(frame.iloc[test_positions]["id_student"].nunique()),
        "shared_students": 0,
        "preprocessor_fit_rows": len(X_train),
    }
    report = {
        "artifact_version": ARTIFACT_VERSION,
        "target": "passed",
        "risk_class": 0,
        "selection_criterion": {
            "primary": "risk_recall",
            "tie_breakers": ["risk_f1", "risk_precision", "roc_auc"],
            "note": "No single metric represents all model quality; guardrails are reported for every candidate.",
        },
        "champion": {"name": champion_name, "configuration": _configuration(champion)},
        "contract": {"version": CONTRACT_VERSION, "fingerprint": contract_fingerprint(), "features": list(FEATURE_COLUMNS)},
        "cutoff_day": cutoff_day,
        "selection": {"strategy": "GroupKFold", "folds": folds, "rows": len(X_train), "holdout_rows_used": 0, "candidates": candidate_rows},
        "evaluation": {"strategy": "GroupShuffleSplit holdout", "rows": len(X_holdout), "champion_metrics": holdout_metrics},
        "split": split,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "model_target": "passed",
        "risk_class": 0,
        "cutoff_day": cutoff_day,
        "champion_name": champion_name,
        "champion_config": _configuration(champion),
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint(),
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "known_key_values": {
            column: sorted(frame.iloc[train_positions][column].dropna().astype(str).unique().tolist())
            for column in ("code_module", "code_presentation")
        },
        "baseline": {
            "numeric_missing_rates": X_train.loc[:, NUMERIC_FEATURES].isna().mean().to_dict(),
            "categorical_missing_rates": X_train.loc[:, CATEGORICAL_FEATURES].isna().mean().to_dict(),
        },
        "training_rows": len(X_train),
        "evaluation_rows": len(X_holdout),
        "model_filename": model_path.name,
        "training_report": report_path.name,
        "metrics": metrics_path.name,
        "evaluation_predictions": predictions_path.name,
    }
    temporary_manifest = artifacts_dir / f".{version}-{MANIFEST_FILENAME}"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return TrainingArtifacts(model_path, manifest_path, report_path, metrics_path, predictions_path)
