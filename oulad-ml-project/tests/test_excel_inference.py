import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oulad_ml_project.inference import (
    FEATURE_COLUMNS,
    build_excel_inference_frame,
    predict_excel,
    train_inference_model,
)


def write_workbook(path: Path, *, missing_registration: bool = False) -> None:
    students = pd.DataFrame(
        {
            "guid_student_id": ["student-1", "student-2"],
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2014J"],
            "gender": ["F", "M"],
            "region": ["North", "New region"],
            "highest_education": ["Bachelor", "Master"],
            "imd_band": ["20-30%", "30-40%"],
            "age_band": ["0-35", "35-55"],
            "num_of_prev_attempts": [0, 1],
            "studied_credits": [60, 90],
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
    courses = pd.DataFrame(
        {"code_module": ["AAA", "BBB"], "code_presentation": ["2013J", "2014J"], "module_presentation_length": [100, 120]}
    )
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

    def test_persisted_model_predicts_with_oov_categories(self):
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
            training_path = root / "training.csv"
            metadata_path = root / "metadata.json"
            workbook = root / "input.xlsx"
            output = root / "predictions.csv"
            training.to_csv(training_path, index=False)
            metadata_path.write_text(json.dumps({"cutoff_day": 30}), encoding="utf-8")
            write_workbook(workbook)
            train_inference_model(training_path, metadata_path, root / "artifacts")
            predict_excel(workbook, root / "artifacts", output, 30)
            predictions = pd.read_csv(output)

        self.assertEqual(len(predictions), 2)
        self.assertIn("region", predictions.loc[1, "oov_categorical_fields"])
        self.assertEqual(predictions.loc[0, "oov_key_fields"], "none")
        self.assertTrue(predictions["probability_passed"].between(0, 1).all())
