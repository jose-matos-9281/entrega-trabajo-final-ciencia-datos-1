"""Leakage-safe Excel inference for the enrollment-level OULAD contract."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_sources import KEY_COLUMNS


ARTIFACT_VERSION = 1
MODEL_FILENAME = "passed_model.joblib"
MANIFEST_FILENAME = "inference_manifest.json"
REQUIRED_SHEETS = ("StudentInfo", "Registration", "VLE_clickStream", "cursos")
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


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _require_complete(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], source: str) -> None:
    missing = [column for column in columns if frame[column].isna().any()]
    if missing:
        raise ValueError(f"{source} has missing values in required columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{source} has duplicate enrollment keys: {columns}")


def _numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source: str,
    *,
    allow_missing: bool = False,
) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not allow_missing:
        _require_complete(frame, columns, source)


def _read_excel_sheets(excel_path: Path) -> dict[str, pd.DataFrame]:
    with pd.ExcelFile(excel_path) as workbook:
        missing_sheets = sorted(set(REQUIRED_SHEETS).difference(workbook.sheet_names))
        if missing_sheets:
            raise ValueError(f"Excel workbook is missing required sheets: {missing_sheets}")
        return {sheet: pd.read_excel(workbook, sheet_name=sheet) for sheet in REQUIRED_SHEETS}


def build_excel_inference_frame(excel_path: Path, cutoff_day: int) -> pd.DataFrame:
    """Map the approved Excel fields to one complete enrollment row per key.

    Assessment sheets are intentionally never loaded: their scores, results, and
    submission dates are post-cutoff outcomes, not inference features.
    """
    if isinstance(cutoff_day, bool) or not isinstance(cutoff_day, int):
        raise TypeError("cutoff_day must be an integer")

    sheets = _read_excel_sheets(excel_path)
    students = sheets["StudentInfo"]
    registrations = sheets["Registration"]
    clicks = sheets["VLE_clickStream"]
    courses = sheets["cursos"]

    student_columns = {
        "guid_student_id", "code_module", "code_presentation", *CATEGORICAL_FEATURES,
        "num_of_prev_attempts", "studied_credits",
    }
    _require_columns(students, student_columns, "StudentInfo")
    students = students.rename(columns={"guid_student_id": "id_student"})[
        [*KEY_COLUMNS, *CATEGORICAL_FEATURES, "num_of_prev_attempts", "studied_credits"]
    ].copy()
    _require_complete(students, KEY_COLUMNS, "StudentInfo")
    _require_unique(students, KEY_COLUMNS, "StudentInfo")
    _numeric(
        students,
        ("num_of_prev_attempts", "studied_credits"),
        "StudentInfo",
        allow_missing=True,
    )

    registration_columns = {
        "guid_studente_id", "code_module", "code_presentation", "date_registration",
    }
    _require_columns(registrations, registration_columns, "Registration")
    registrations = registrations.rename(columns={"guid_studente_id": "id_student"})[
        [*KEY_COLUMNS, "date_registration"]
    ].copy()
    _require_complete(registrations, KEY_COLUMNS, "Registration")
    _require_unique(registrations, KEY_COLUMNS, "Registration")
    _numeric(registrations, ("date_registration",), "Registration")
    if (registrations["date_registration"] > cutoff_day).any():
        raise ValueError("Registration includes enrollments that start after cutoff_day")

    course_columns = {"code_module", "code_presentation", "module_presentation_length"}
    _require_columns(courses, course_columns, "cursos")
    courses = courses.rename(columns={"module_presentation_length": "course_duration_days"})[
        ["code_module", "code_presentation", "course_duration_days"]
    ].copy()
    _require_complete(courses, ("code_module", "code_presentation", "course_duration_days"), "cursos")
    _require_unique(courses, ["code_module", "code_presentation"], "cursos")
    _numeric(courses, ("course_duration_days",), "cursos")
    if (courses["course_duration_days"] <= 0).any():
        raise ValueError("cursos has non-positive course_duration_days")

    click_columns = {
        "guid_student_id", "guid_site_id", "modulo", "presentation", "date", "sum_clics",
    }
    _require_columns(clicks, click_columns, "VLE_clickStream")
    clicks = clicks.rename(
        columns={
            "guid_student_id": "id_student",
            "guid_site_id": "id_site",
            "modulo": "code_module",
            "presentation": "code_presentation",
            "sum_clics": "clicks",
        }
    )[[*KEY_COLUMNS, "id_site", "date", "clicks"]].copy()
    _require_complete(clicks, ("id_student", "code_presentation", "id_site", "date", "clicks"), "VLE_clickStream")

    _numeric(clicks, ("date", "clicks"), "VLE_clickStream")
    if (clicks["clicks"] < 0).any():
        raise ValueError("VLE_clickStream has negative click counts")

    # Events after the cutoff cannot affect inference features, so discard them
    # before resolving their optional source module values.
    pre_cutoff_clicks = clicks.loc[clicks["date"] <= cutoff_day].copy()
    enrollment_modules = students[["id_student", "code_presentation", "code_module"]]
    blank_module = pre_cutoff_clicks["code_module"].isna()
    if blank_module.any():
        module_counts = enrollment_modules.groupby(["id_student", "code_presentation"])["code_module"].size()
        blank_keys = pd.MultiIndex.from_frame(pre_cutoff_clicks.loc[blank_module, ["id_student", "code_presentation"]])
        candidate_counts = module_counts.reindex(blank_keys)
        no_candidate = candidate_counts.isna()
        ambiguous_candidate = candidate_counts.gt(1)
        discard = no_candidate | ambiguous_candidate
        if discard.any():
            reasons = []
            if no_candidate.any():
                reasons.append(f"{no_candidate.sum()} without an enrollment candidate")
            if ambiguous_candidate.any():
                reasons.append(f"{ambiguous_candidate.sum()} with ambiguous enrollment candidates")
            warnings.warn(
                f"Discarded {discard.sum()} pre-cutoff VLE_clickStream events: {'; '.join(reasons)}.",
                UserWarning,
                stacklevel=2,
            )
            pre_cutoff_clicks = pre_cutoff_clicks.drop(index=pre_cutoff_clicks.index[blank_module][discard.to_numpy()])
            blank_module = pre_cutoff_clicks["code_module"].isna()
        resolved = pre_cutoff_clicks.loc[blank_module, ["id_student", "code_presentation"]].merge(
            enrollment_modules.drop_duplicates(["id_student", "code_presentation"]),
            on=["id_student", "code_presentation"],
            how="left",
            validate="many_to_one",
        )
        pre_cutoff_clicks.loc[blank_module, "code_module"] = resolved["code_module"].to_numpy()
    _require_complete(pre_cutoff_clicks, KEY_COLUMNS, "VLE_clickStream")

    enrollment = students.merge(registrations, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(enrollment) != len(students) or len(enrollment) != len(registrations):
        raise ValueError("StudentInfo and Registration do not have identical enrollment coverage")
    enrollment = enrollment.merge(
        courses, on=["code_module", "code_presentation"], how="left", validate="many_to_one"
    )
    _require_complete(enrollment, ("course_duration_days",), "cursos coverage")

    if not pre_cutoff_clicks.empty:
        activity = pre_cutoff_clicks.groupby(KEY_COLUMNS, as_index=False).agg(
            total_clicks=("clicks", "sum"),
            active_days=("date", "nunique"),
            vle_events=("date", "size"),
            vle_sites=("id_site", "nunique"),
        )
        covered = activity.merge(enrollment[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
        if (covered["_merge"] != "both").any():
            raise ValueError("VLE_clickStream contains activity without a matching enrollment")
    else:
        activity = pd.DataFrame(columns=[*KEY_COLUMNS, "total_clicks", "active_days", "vle_events", "vle_sites"])

    frame = enrollment.merge(activity, on=KEY_COLUMNS, how="left", validate="one_to_one")
    activity_columns = ["total_clicks", "active_days", "vle_events", "vle_sites"]
    frame[activity_columns] = frame[activity_columns].fillna(0)
    frame["has_vle_activity"] = (frame["vle_events"] > 0).astype(int)
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Excel mapping did not produce one row per enrollment")
    return frame[[*KEY_COLUMNS, *FEATURE_COLUMNS]].sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def train_inference_model(training_data: Path, metadata_path: Path, artifacts_dir: Path) -> dict[str, Path]:
    """Fit and persist the preprocessing pipeline and binary passed model."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cutoff_day = metadata.get("cutoff_day")
    if isinstance(cutoff_day, bool) or not isinstance(cutoff_day, int):
        raise ValueError("Training metadata must contain an integer cutoff_day")
    frame = pd.read_csv(training_data)
    _require_columns(frame, set(KEY_COLUMNS) | set(FEATURE_COLUMNS) | {"passed"}, "training data")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("training data has duplicate enrollment keys")
    if frame["passed"].isna().any() or frame["passed"].nunique() != 2:
        raise ValueError("training data must contain both non-null passed classes")

    training_features = frame[list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_FEATURES:
        training_features[column] = pd.to_numeric(training_features[column], errors="coerce")
        if training_features[column].isna().all():
            raise ValueError(f"training data has no usable values for {column}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]
    )
    model.fit(training_features, frame["passed"].astype(int))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_FILENAME
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    joblib.dump(model, model_path)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "cutoff_day": cutoff_day,
        "model_target": "passed",
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "known_key_values": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in ("code_module", "code_presentation")
        },
        "training_rows": len(frame),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": model_path, "manifest": manifest_path}


def _oov_columns(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    categorical_transformer = model.named_steps["preprocessor"].named_transformers_["categorical"]
    encoder = categorical_transformer.named_steps["encoder"]
    known = dict(zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True))
    values = pd.DataFrame(
        {column: ~features[column].astype(str).isin(set(categories.astype(str))) for column, categories in known.items()}
    )
    return values.apply(lambda row: ";".join(row.index[row].tolist()) or "none", axis=1)


def _oov_key_columns(frame: pd.DataFrame, manifest: dict[str, object]) -> pd.Series:
    known_values = manifest.get("known_key_values")
    if not isinstance(known_values, dict):
        raise ValueError("Inference artifact is missing known enrollment key values")
    values = pd.DataFrame(
        {
            column: ~frame[column].astype(str).isin(set(known_values.get(column, [])))
            for column in ("code_module", "code_presentation")
        }
    )
    return values.apply(lambda row: ";".join(row.index[row].tolist()) or "none", axis=1)


def predict_excel(excel_path: Path, artifacts_dir: Path, output_path: Path, cutoff_day: int) -> Path:
    """Generate enrollment-level predictions without mutating the source workbook."""
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    model_path = artifacts_dir / MODEL_FILENAME
    if not manifest_path.exists() or not model_path.exists():
        raise FileNotFoundError("Inference artifacts are incomplete; run train-inference-model first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cutoff_day") != cutoff_day:
        raise ValueError("cutoff_day must exactly match the trained inference artifact")
    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Inference artifact feature contract is incompatible")

    frame = build_excel_inference_frame(excel_path, cutoff_day)
    model = joblib.load(model_path)
    features = frame[list(FEATURE_COLUMNS)]
    output = frame[KEY_COLUMNS].copy()
    output["cutoff_day"] = cutoff_day
    output["model_target"] = manifest["model_target"]
    output["prediction_passed"] = model.predict(features).astype(int)
    output["probability_passed"] = model.predict_proba(features)[:, 1]
    output["oov_categorical_fields"] = _oov_columns(model, features)
    output["oov_key_fields"] = _oov_key_columns(frame, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == excel_path.resolve():
        raise ValueError("Prediction output must not overwrite the source workbook")
    output.to_csv(output_path, index=False)
    return output_path


def main_train() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train persisted OULAD Excel inference artifacts.")
    parser.add_argument("--training-data", type=Path, default=project_dir / "data" / "oulad_kongo_full.csv")
    parser.add_argument("--training-metadata", type=Path, default=project_dir / "data" / "oulad_training_metadata.json")
    parser.add_argument("--artifacts-dir", type=Path, default=project_dir / "artifacts")
    args = parser.parse_args()
    paths = train_inference_model(args.training_data, args.training_metadata, args.artifacts_dir)
    print(f"Persisted inference model and manifest in {args.artifacts_dir}")


def main_predict() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Predict OULAD enrollments from the supported Excel workbook.")
    parser.add_argument("--excel", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-day", type=int, required=True)
    args = parser.parse_args()
    path = predict_excel(args.excel, args.artifacts_dir, args.output, args.cutoff_day)
    print(f"Wrote predictions to {path}")
