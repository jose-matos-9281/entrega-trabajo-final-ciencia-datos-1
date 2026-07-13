import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from oulad_ml_project.data_sources import KEY_COLUMNS, TARGET_COLUMNS
from oulad_ml_project.eda import MISSING_IMD_LABEL, ExploratoryDataAnalysis, main


def sample_mart() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(
            {
                "id_student": index + 1,
                "code_module": "AAA" if index < 4 else "BBB",
                "code_presentation": "2013J" if index % 2 == 0 else "2014J",
                "gender": "F" if index % 2 else "M",
                "region": "North",
                "highest_education": "Bachelor",
                "imd_band": None if index == 0 else "20-30%",
                "disability": False,
                "age_band": "0-35",
                "num_of_prev_attempts": index % 3,
                "studied_credits": 60,
                "date_registration": -10 + index,
                "course_duration_days": 250,
                "total_clicks": index * 10,
                "active_days": index + 1,
                "vle_events": index * 2,
                "vle_sites": index + 1,
                "has_vle_activity": int(index > 0),
                "final_result": "Pass" if index >= 4 else "Fail",
                "passed": int(index >= 4),
                "performance_tier": index % 4,
                "weighted_assessment_score": None if index in {0, 1, 4} else 50 + index,
            }
        )
    return pd.DataFrame(rows)


class ExploratoryDataAnalysisTests(unittest.TestCase):
    def test_report_and_figure_manifest_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            manifest = eda.run_all()
            output = Path(directory) / "output"

            self.assertTrue((output / "eda_report.md").exists())
            self.assertTrue((output / "figures" / "cohort_engagement_total_clicks.png").exists())
            self.assertTrue((output / "final_result_distribution.csv").exists())
            self.assertTrue((output / "academic_risk_by_engagement_quartile.csv").exists())
            self.assertTrue((output / "score_missingness_by_final_result.csv").exists())
            self.assertTrue((output / "figures" / "final_result_distribution.png").exists())
            self.assertTrue((output / "figures" / "score_missingness_by_final_result.png").exists())
            self.assertIn(output / "eda_report.md", manifest)
            report = (output / "eda_report.md").read_text(encoding="utf-8")
            self.assertIn("figures/academic_risk_by_engagement_quartile.png", report)
            self.assertIn("Fail: 4/8", report)
            self.assertIn("# EDA de OULAD: alerta temprana", report)
            self.assertIn("no demuestran causalidad", report)
            self.assertIn(f"etiqueta `{MISSING_IMD_LABEL}`", report)
            self.assertNotIn("`Missing`", report)

    def test_chart_labels_are_in_neutral_professional_spanish(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("matplotlib.axes.Axes.set_title") as set_title, \
             patch("matplotlib.axes.Axes.set_xlabel") as set_xlabel, \
             patch("matplotlib.axes.Axes.set_ylabel") as set_ylabel:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            eda.run_all()

        titles = [call.args[0] for call in set_title.call_args_list]
        x_labels = [call.args[0] for call in set_xlabel.call_args_list]
        y_labels = [call.args[0] for call in set_ylabel.call_args_list]
        self.assertIn("Interacción de la cohorte: total de clics antes del corte", titles)
        self.assertIn("Valores faltantes y disponibilidad de puntuaciones observadas", titles)
        self.assertIn("Variable", x_labels)
        self.assertIn("Tasa de riesgo académico (%)", y_labels)
        self.assertFalse(any("Pass Rate" in label or "Missingness" in label for label in titles))

    def test_target_columns_are_excluded_from_candidate_correlations(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            correlation = eda.candidate_spearman_correlations()

        self.assertTrue(set(TARGET_COLUMNS).isdisjoint(correlation.columns))
        self.assertTrue(set(KEY_COLUMNS).isdisjoint(correlation.columns))

    def test_academic_risk_uses_descriptive_outcomes_and_keeps_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            final_result, risk, engagement, _, score_missingness = eda.academic_risk_analysis()

        self.assertEqual(engagement["n_total"].sum(), 8)
        self.assertEqual(engagement["n_academic_risk"].sum(), 4)
        self.assertEqual(engagement.iloc[0]["academic_risk_rate"], 1.0)
        self.assertEqual(engagement.iloc[-1]["academic_risk_rate"], 0.0)
        self.assertEqual(final_result.set_index("final_result").loc["Fail", "n_total"], 4)
        self.assertEqual(risk.set_index("academic_risk").loc["Riesgo académico", "n_total"], 4)
        self.assertEqual(score_missingness.set_index("final_result").loc["Fail", "n_score_missing"], 2)
        self.assertNotIn("academic_risk", eda.candidate_features)

    def test_sparse_course_groups_are_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            _, _, _, course, _ = eda.academic_risk_analysis()

        self.assertTrue(course["suppressed"].all())
        self.assertTrue(course["academic_risk_rate"].isna().all())

    def test_duplicate_enrollment_key_is_rejected(self):
        frame = pd.concat([sample_mart(), sample_mart().iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(frame, Path(directory), Path(directory) / "output")
            with self.assertRaisesRegex(ValueError, "una fila por clave de matrícula"):
                eda.run_all()

    def test_cli_writes_report_for_input_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_csv = root / "mart.csv"
            output = root / "eda-output"
            sample_mart().to_csv(input_csv, index=False)
            with patch.object(sys, "argv", ["eda", "--input-csv", str(input_csv), "--output-dir", str(output)]):
                main()

            self.assertTrue((output / "eda_report.md").exists())


if __name__ == "__main__":
    unittest.main()
