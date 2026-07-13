"""Leakage-safe academic-risk training and artifact reporting for OULAD."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_sources import KEY_COLUMNS
from .feature_contract import (
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OUTCOME_COLUMNS,
    SENSITIVE_COLUMNS,
    contract_fingerprint,
    validate_raw_features,
)


ARTIFACT_VERSION = 3
MODEL_FILENAME = "academic_risk_model.joblib"
MANIFEST_FILENAME = "inference_manifest.json"
TRAINING_REPORT_FILENAME = "training_report.json"
METRICS_FILENAME = "training_metrics.csv"
EVALUATION_PREDICTIONS_FILENAME = "evaluation_predictions.csv"
TARGET_DEFINITION = "academic_risk = 1 when final_result is Withdrawn or Fail; 0 when Pass or Distinction"
RISK_RESULTS = {"Withdrawn", "Fail"}
NON_RISK_RESULTS = {"Pass", "Distinction"}


@dataclass(frozen=True)
class TrainingArtifacts:
    model: Path
    manifest: Path
    report: Path
    metrics: Path
    evaluation_predictions: Path

    def as_dict(self) -> dict[str, Path]:
        return {"model": self.model, "manifest": self.manifest, "report": self.report,
                "metrics": self.metrics, "evaluation_predictions": self.evaluation_predictions}


def _pipeline(classifier: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                         ("encoder", OneHotEncoder(handle_unknown="ignore"))]), list(CATEGORICAL_FEATURES)),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                     ("scaler", StandardScaler())]), list(NUMERIC_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _candidates() -> dict[str, Pipeline]:
    """Keep comparison limited to a baseline, an interpretable linear model, and one bounded tree."""
    return {
        "PrevalenceBaseline": _pipeline(DummyClassifier(strategy="prior")),
        "LogisticRegression": _pipeline(LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        "DecisionTreeClassifier": _pipeline(DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)),
    }


def _metrics(y_true: pd.Series, probability_risk: pd.Series) -> dict[str, float | None]:
    threshold_predictions = (probability_risk >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probability_risk)),
        "roc_auc": float(roc_auc_score(y_true, probability_risk)) if y_true.nunique() == 2 else None,
        "brier_score": float(brier_score_loss(y_true, probability_risk)),
        "precision_at_0_5": float(precision_score(y_true, threshold_predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, threshold_predictions, zero_division=0)),
    }


def _calibration(y_true: pd.Series, probability_risk: pd.Series) -> list[dict[str, float | int]]:
    observed, predicted = calibration_curve(y_true, probability_risk, n_bins=5, strategy="quantile")
    bins = pd.qcut(probability_risk.rank(method="first"), q=len(observed), duplicates="drop")
    counts = bins.value_counts(sort=False).to_numpy()
    return [{"mean_predicted_risk": float(pred), "observed_risk": float(obs), "rows": int(count)}
            for pred, obs, count in zip(predicted, observed, counts, strict=True)]


def _capacity_metrics(y_true: pd.Series, probability_risk: pd.Series, capacity: float = 0.20) -> dict[str, float | int]:
    selected = max(1, int(len(y_true) * capacity))
    intervention = probability_risk.sort_values(ascending=False, kind="stable").index[:selected]
    predicted = pd.Series(0, index=y_true.index)
    predicted.loc[intervention] = 1
    return {"capacity": capacity, "selected_rows": selected,
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0))}


def _configuration(model: Pipeline) -> dict[str, object]:
    classifier = model.named_steps["classifier"]
    return {"classifier": classifier.__class__.__name__, "parameters": classifier.get_params(),
            "preprocessor": "fold/training-only categorical most-frequent + one-hot; numeric median + scaling"}


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    return {key: (float(values.mean()) if values.notna().any() else None)
            for key in rows[0] if (values := pd.Series([row[key] for row in rows])) is not None}


def _target(frame: pd.DataFrame) -> pd.Series:
    if "final_result" not in frame.columns:
        raise ValueError("training data must contain final_result to derive academic_risk")
    unexpected = sorted(set(frame["final_result"].dropna()).difference(RISK_RESULTS | NON_RISK_RESULTS))
    if frame["final_result"].isna().any() or unexpected:
        raise ValueError("final_result must contain only Withdrawn, Fail, Pass, or Distinction")
    return frame["final_result"].isin(RISK_RESULTS).astype(int)


def _temporal_split(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.Index, pd.Index, dict[str, object]]:
    if "2014J" in set(frame["code_presentation"]):
        test_positions = frame.index[frame["code_presentation"] == "2014J"]
        test_students = set(frame.loc[test_positions, "id_student"])
        development = frame.index[(frame["code_presentation"] != "2014J") & ~frame["id_student"].isin(test_students)]
        strategy = "temporal_holdout_2014J"
        removed_overlap = int(((frame["code_presentation"] != "2014J") & frame["id_student"].isin(test_students)).sum())
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        development_positions, test_positions_array = next(splitter.split(frame, target, frame["id_student"]))
        development = frame.index[development_positions]
        test_positions = frame.index[test_positions_array]
        strategy = "grouped_random_holdout_fallback"
        removed_overlap = 0
    if target.loc[development].nunique() != 2 or target.loc[test_positions].nunique() != 2:
        raise ValueError("development and holdout partitions must each contain both academic_risk classes")
    return development, test_positions, {"strategy": strategy, "holdout_presentation": "2014J" if strategy.startswith("temporal") else None,
                                          "overlap_rows_removed_from_development": removed_overlap}


def train_risk_champion(frame: pd.DataFrame, artifacts_dir: Path, cutoff_day: int, *, data_filename: str = "oulad_training_full.csv") -> TrainingArtifacts:
    """Select by grouped development CV using PR-AUC, then evaluate once on a temporal holdout."""
    if isinstance(cutoff_day, bool) or not isinstance(cutoff_day, int):
        raise TypeError("cutoff_day must be an integer")
    features = validate_raw_features(frame, "training data", allow_outcomes=True)
    target = _target(frame)
    development, holdout, split_details = _temporal_split(frame, target)
    X_development, X_holdout = features.loc[development], features.loc[holdout]
    y_development, y_holdout = target.loc[development], target.loc[holdout]
    development_groups = frame.loc[development, "id_student"]
    folds = min(5, development_groups.nunique())
    if folds < 2:
        raise ValueError("development partition needs at least two students for grouped CV")

    candidate_rows: list[dict[str, object]] = []
    cv = GroupKFold(n_splits=folds)
    for name, candidate in _candidates().items():
        fold_metrics = []
        for fit_positions, validation_positions in cv.split(X_development, y_development, development_groups):
            model = clone(candidate).fit(X_development.iloc[fit_positions], y_development.iloc[fit_positions])
            probability = pd.Series(model.predict_proba(X_development.iloc[validation_positions])[:, 1], index=y_development.iloc[validation_positions].index)
            fold_metrics.append(_metrics(y_development.iloc[validation_positions], probability))
        candidate_rows.append({"model": name, **_mean_metrics(fold_metrics)})

    champion_name = max(candidate_rows, key=lambda row: (float(row["pr_auc"] or 0), -(float(row["brier_score"] or 1))))["model"]
    champion = _candidates()[str(champion_name)].fit(X_development, y_development)
    probability_risk = pd.Series(champion.predict_proba(X_holdout)[:, 1], index=X_holdout.index)
    holdout_metrics = _metrics(y_holdout, probability_risk)
    holdout_metrics["intervention_capacity"] = _capacity_metrics(y_holdout, probability_risk)
    holdout_calibration = _calibration(y_holdout, probability_risk)
    evaluation_rows = pd.DataFrame({**{column: frame.loc[holdout, column].to_numpy() for column in KEY_COLUMNS},
                                    "model": champion_name, "academic_risk_actual": y_holdout.to_numpy(),
                                    "probability_academic_risk": probability_risk.to_numpy(),
                                    "prediction_academic_risk": (probability_risk >= 0.5).astype(int).to_numpy()})

    version = uuid4().hex
    bundle_dir = artifacts_dir / version
    bundle_dir.mkdir(parents=True, exist_ok=False)
    model_path = bundle_dir / MODEL_FILENAME
    manifest_path = bundle_dir / MANIFEST_FILENAME
    report_path = bundle_dir / TRAINING_REPORT_FILENAME
    metrics_path = bundle_dir / METRICS_FILENAME
    predictions_path = bundle_dir / EVALUATION_PREDICTIONS_FILENAME
    joblib.dump(champion, model_path)
    pd.DataFrame(candidate_rows).to_csv(metrics_path, index=False)
    evaluation_rows.to_csv(predictions_path, index=False)

    exclusions = {"outcomes": list(OUTCOME_COLUMNS), "enrollment_keys": list(KEY_COLUMNS),
                  "sensitive_predictors": list(SENSITIVE_COLUMNS), "post_cutoff": "unsupported source columns are rejected"}
    split = {**split_details, "group_column": "id_student", "train_rows": len(X_development), "test_rows": len(X_holdout),
             "train_students": int(frame.loc[development, "id_student"].nunique()), "test_students": int(frame.loc[holdout, "id_student"].nunique()),
             "shared_students": 0, "preprocessor_fit_rows": len(X_development)}
    report = {"artifact_version": ARTIFACT_VERSION, "data_filename": data_filename, "cutoff_day": cutoff_day,
              "target": {"name": "academic_risk", "definition": TARGET_DEFINITION, "positive_class": 1},
              "feature_exclusions": exclusions, "champion": {"name": champion_name, "configuration": _configuration(champion)},
              "contract": {"version": CONTRACT_VERSION, "fingerprint": contract_fingerprint(), "features": list(FEATURE_COLUMNS)},
              "selection": {"strategy": "GroupKFold", "group_column": "id_student", "folds": folds, "primary_metric": "pr_auc", "rows": len(X_development), "holdout_rows_used": 0, "candidates": candidate_rows},
              "evaluation": {"strategy": split_details["strategy"], "rows": len(X_holdout), "champion_metrics": holdout_metrics, "calibration": holdout_calibration},
              "split": split, "limitations": ["Associations are predictive, not causal.", "The temporal holdout is evaluated once after model selection."]}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {"artifact_version": ARTIFACT_VERSION, "data_filename": data_filename, "model_target": "academic_risk", "target_definition": TARGET_DEFINITION,
                "cutoff_day": cutoff_day, "champion_name": champion_name, "champion_config": _configuration(champion),
                "contract_version": CONTRACT_VERSION, "contract_fingerprint": contract_fingerprint(), "feature_columns": list(FEATURE_COLUMNS),
                "categorical_features": list(CATEGORICAL_FEATURES), "feature_exclusions": exclusions,
                "known_key_values": {column: sorted(frame.loc[development, column].dropna().astype(str).unique().tolist()) for column in ("code_module", "code_presentation")},
                "baseline": {"numeric_missing_rates": X_development.loc[:, NUMERIC_FEATURES].isna().mean().to_dict(), "categorical_missing_rates": X_development.loc[:, CATEGORICAL_FEATURES].isna().mean().to_dict()},
                 "split_strategy": split, "model_metrics": {"selection": candidate_rows, "holdout": holdout_metrics},
                 "training_rows": len(X_development), "evaluation_rows": len(X_holdout),
                  "model_path": model_path.name,
                  "training_report": report_path.name,
                  "metrics": metrics_path.name,
                  "evaluation_predictions": predictions_path.name}
    temporary_manifest = bundle_dir / f".{MANIFEST_FILENAME}"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return TrainingArtifacts(model_path, manifest_path, report_path, metrics_path, predictions_path)
