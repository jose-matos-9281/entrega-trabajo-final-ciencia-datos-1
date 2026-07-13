import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from oulad_ml_project.feature_contract import CONTRACT_VERSION, NUMERIC_FEATURES
from oulad_ml_project.inference import (
    FEATURE_COLUMNS,
    build_excel_inference_frame,
    predict_excel,
    train_inference_model,
)


def write_workbook(
    path: Path,
    *,
    missing_registration: bool = False,
    missing_optional_features: bool = False,
    unresolved_vle_date: int | None = None,
    ambiguous_vle_module: bool = False,
) -> None:
    students = pd.DataFrame(
        {
            "guid_student_id": ["student-1", "student-2"],
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2014J"],
            "gender": ["F", "M"],
            "region": ["North", "New region"],
            "highest_education": ["Bachelor", "Master"],
            "imd_band": ["20-30%", None if missing_optional_features else "30-40%"],
            "age_band": [None if missing_optional_features else "0-35", "35-55"],
            "num_of_prev_attempts": [0, None if missing_optional_features else 1],
            "studied_credits": [None if missing_optional_features else 60, 90],
            "disability": ["N", "N"],
            "final_result": ["Pass", "Fail"],
        }
    )
    registrations = pd.DataFrame(
        {
            "guid_studente_id": ["student-1"] if missing_registration else ["student-1", "student-2"],
            "code_module": ["AAA"] if missing_registration else ["AAA", "BBB"],
            "code_presentation": ["2013J"] if missing_registration else ["2013J", "2014J"],
            "date_registration": [-10] if missing_registration else [-10, -5],
            "date_unregistration": [None] if missing_registration else [None, None],
        }
    )
    if ambiguous_vle_module:
        students.loc[len(students)] = ["student-2", "CCC", "2014J", "M", "North", "Master", "30-40%", "35-55", 1, 90, "N", "Pass"]
        registrations.loc[len(registrations)] = ["student-2", "CCC", "2014J", -5, None]
    clicks = pd.DataFrame(
        {
            "guid_student_id": ["student-1", "student-2"],
            "guid_site_id": [10, 20],
            "modulo": ["AAA", None],
            "presentation": ["2013J", "2014J"],
            "date": [5, 10],
            "sum_clics": [4, 7],
        }
    )
    if unresolved_vle_date is not None:
        clicks.loc[len(clicks)] = ["unknown-student", 30, None, "2024J", unresolved_vle_date, 5]
    courses = pd.DataFrame(
        {"code_module": ["AAA", "BBB"], "code_presentation": ["2013J", "2014J"], "module_presentation_length": [100, 120]}
    )
    if ambiguous_vle_module:
        courses.loc[len(courses)] = ["CCC", "2014J", 120]
    assessments = pd.DataFrame({"score": [100], "final_result": ["Pass"], "date_submitted": [90]})
    with pd.ExcelWriter(path) as writer:
        students.to_excel(writer, sheet_name="StudentInfo", index=False)
        registrations.to_excel(writer, sheet_name="Registration", index=False)
        clicks.to_excel(writer, sheet_name="VLE_clickStream", index=False)
        courses.to_excel(writer, sheet_name="cursos", index=False)
        assessments.to_excel(writer, sheet_name="Assesss_detail", index=False)


class ExcelInferenceTest(unittest.TestCase):
    def test_adapter_uses_only_allowed_fields_and_cutoff_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "input.xlsx"
            write_workbook(workbook)
            frame = build_excel_inference_frame(workbook, 30)

        self.assertEqual(frame.columns.tolist(), ["id_student", "code_module", "code_presentation", *FEATURE_COLUMNS])
        self.assertEqual(frame["total_clicks"].tolist(), [4, 7])
        self.assertEqual(frame["vle_sites"].tolist(), [1, 1])
        self.assertNotIn("score", frame.columns)
        self.assertNotIn("final_result", frame.columns)

    def test_adapter_rejects_incomplete_enrollment_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "input.xlsx"
            write_workbook(workbook, missing_registration=True)
            with self.assertRaisesRegex(ValueError, "identical enrollment coverage"):
                build_excel_inference_frame(workbook, 30)

    def test_adapter_ignores_unresolved_module_activity_after_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "input.xlsx"
            write_workbook(workbook, unresolved_vle_date=31)
            frame = build_excel_inference_frame(workbook, 30)

        self.assertEqual(frame["total_clicks"].tolist(), [4, 7])

    def test_adapter_discards_unresolved_module_activity_before_cutoff_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "input.xlsx"
            write_workbook(workbook, unresolved_vle_date=30)
            with self.assertWarnsRegex(
                UserWarning,
                r"Discarded 1 pre-cutoff VLE_clickStream events: 1 without an enrollment candidate\.",
            ):
                frame = build_excel_inference_frame(workbook, 30)

        self.assertEqual(frame["total_clicks"].tolist(), [4, 7])

    def test_adapter_discards_ambiguous_module_activity_before_cutoff_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "input.xlsx"
            write_workbook(workbook, ambiguous_vle_module=True)
            with self.assertWarnsRegex(
                UserWarning,
                r"Discarded 1 pre-cutoff VLE_clickStream events: 1 with ambiguous enrollment candidates\.",
            ):
                frame = build_excel_inference_frame(workbook, 30)

        self.assertEqual(frame["total_clicks"].tolist(), [4, 0, 0])

    def test_persisted_model_predicts_with_oov_and_missing_optional_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = pd.DataFrame(
                {
                    "id_student": [1, 2, 3, 4],
                    "code_module": ["AAA", "AAA", "BBB", "BBB"],
                    "code_presentation": ["2013J", "2013J", "2014J", "2014J"],
                    "gender": ["F", "M", "F", "M"],
                    "region": ["North", "South", "North", "South"],
                    "highest_education": ["Bachelor", "Master", "Bachelor", "Master"],
                    "imd_band": ["20-30%", "30-40%", "20-30%", "30-40%"],
                    "disability": ["N", "N", "N", "N"],
                    "age_band": ["0-35", "35-55", "0-35", "35-55"],
                    "num_of_prev_attempts": [0, 1, 0, 1],
                    "studied_credits": [60, 90, 60, 90],
                    "date_registration": [-10, -5, -10, -5],
                    "course_duration_days": [100, 100, 120, 120],
                    "total_clicks": [4, 7, 5, 8],
                    "active_days": [1, 1, 1, 1],
                    "vle_events": [1, 1, 1, 1],
                    "vle_sites": [1, 1, 1, 1],
                    "has_vle_activity": [1, 1, 1, 1],
                    "passed": [0, 1, 0, 1],
                }
            )
            training = pd.concat([training, training.assign(id_student=lambda rows: rows["id_student"] + 4)], ignore_index=True)
            training.loc[5, "passed"] = 0  # GroupShuffleSplit's holdout must contain both classes.
            training_path = root / "training.csv"
            metadata_path = root / "metadata.json"
            workbook = root / "input.xlsx"
            output = root / "predictions.csv"
            training.to_csv(training_path, index=False)
            metadata_path.write_text(json.dumps({"cutoff_day": 30}), encoding="utf-8")
            write_workbook(workbook, missing_optional_features=True)
            train_inference_model(training_path, metadata_path, root / "artifacts")
            frame = build_excel_inference_frame(workbook, 30)
            predict_excel(workbook, root / "artifacts", output, 30)
            predictions = pd.read_csv(output)
            report = json.loads(output.with_suffix(".inference_report.json").read_text(encoding="utf-8"))

        self.assertTrue(
            frame[["imd_band", "age_band", "num_of_prev_attempts", "studied_credits"]].isna().any().all()
        )
        self.assertEqual(len(predictions), 2)
        self.assertIn("region", predictions.loc[1, "oov_categorical_fields"])
        self.assertEqual(predictions.loc[0, "oov_key_fields"], "none")
        self.assertTrue(predictions["probability_passed"].between(0, 1).all())
        self.assertEqual(report["contract_version"], CONTRACT_VERSION)
        self.assertEqual(report["rows"], 2)

    def test_champion_bundle_uses_train_cv_and_separate_holdout_evaluation(self):
        rows = []
        for student in range(1, 17):
            rows.append(
                {
                    "id_student": student,
                    "code_module": "AAA" if student % 2 else "BBB",
                    "code_presentation": "2013J" if student % 2 else "2014J",
                    "gender": "F" if student % 2 else "M",
                    "region": "North" if student % 3 else "South",
                    "highest_education": "Bachelor",
                    "imd_band": "20-30%",
                    "disability": "N",
                    "age_band": "0-35",
                    "num_of_prev_attempts": student % 3,
                    "studied_credits": 60 + student,
                    "date_registration": -student,
                    "course_duration_days": 100,
                    "total_clicks": student * 10,
                    "active_days": student,
                    "vle_events": student,
                    "vle_sites": 1,
                    "has_vle_activity": 1,
                    "passed": student % 2,
                }
            )
        training = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.csv"
            metadata_path = root / "metadata.json"
            workbook = root / "input.xlsx"
            output = root / "predictions.csv"
            training.to_csv(training_path, index=False)
            metadata_path.write_text(json.dumps({"cutoff_day": 30}), encoding="utf-8")
            write_workbook(workbook)
            paths = train_inference_model(training_path, metadata_path, root / "artifacts")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            bundle = joblib.load(paths["model"])
            predict_excel(workbook, root / "artifacts", output, 30)
            inference_report = json.loads(output.with_suffix(".inference_report.json").read_text(encoding="utf-8"))

        _, test_positions = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
                training, training["passed"], training["id_student"]
            )
        )
        train_positions = training.index.difference(training.index[test_positions])
        imputer = bundle.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
        total_clicks_position = list(NUMERIC_FEATURES).index("total_clicks")

        self.assertEqual(manifest["champion_name"], report["champion"]["name"])
        self.assertEqual(bundle.named_steps["classifier"].__class__.__name__, manifest["champion_name"])
        self.assertEqual(manifest["training_rows"], len(train_positions))
        self.assertEqual(report["selection"]["strategy"], "GroupKFold")
        self.assertEqual(report["selection"]["rows"], len(train_positions))
        self.assertEqual(report["selection"]["holdout_rows_used"], 0)
        self.assertEqual(report["evaluation"]["strategy"], "GroupShuffleSplit holdout")
        self.assertEqual(report["evaluation"]["rows"], len(test_positions))
        self.assertIn("champion_metrics", report["evaluation"])
        self.assertAlmostEqual(
            imputer.statistics_[total_clicks_position], training.loc[train_positions, "total_clicks"].median()
        )
        self.assertEqual(inference_report["champion_name"], manifest["champion_name"])
        self.assertEqual(inference_report["contract_version"], CONTRACT_VERSION)
