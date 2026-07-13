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
            self.assertTrue((output / "total_clicks_by_final_result_boxplot_summary.csv").exists())
            self.assertTrue((output / "bivariate_score_boxplots_summary.csv").exists())
            self.assertTrue((output / "academic_risk_by_engagement_quartile.csv").exists())
            self.assertTrue((output / "score_missingness_by_final_result.csv").exists())
            self.assertTrue((output / "figures" / "final_result_distribution.png").exists())
            self.assertTrue((output / "figures" / "total_clicks_by_final_result_boxplot.png").exists())
            self.assertTrue((output / "figures" / "bivariate_score_boxplots.png").exists())
            self.assertTrue((output / "figures" / "score_missingness_by_final_result.png").exists())
            self.assertTrue((output / "figures" / "univariate_numeric_histograms.png").exists())
            self.assertTrue((output / "figures" / "univariate_categorical_frequencies.png").exists())
            self.assertTrue((output / "univariate_numeric_summary.csv").exists())
            self.assertTrue((output / "univariate_categorical_summary.csv").exists())
            self.assertIn(output / "eda_report.md", manifest)
            report = (output / "eda_report.md").read_text(encoding="utf-8")
            self.assertIn("figures/academic_risk_by_engagement_quartile.png", report)
            self.assertIn("total_clicks_by_final_result_boxplot_summary.csv", report)
            self.assertIn("figures/total_clicks_by_final_result_boxplot.png", report)
            self.assertIn("bivariate_score_boxplots_summary.csv", report)
            self.assertIn("figures/bivariate_score_boxplots.png", report)
            self.assertIn("Fail: 4/8", report)
            self.assertIn("# EDA de OULAD: alerta temprana", report)
            self.assertIn("no demuestran causalidad", report)
            self.assertIn("Distribuciones univariadas de variables candidatas", report)
            self.assertIn("univariate_numeric_summary.csv", report)
            self.assertIn("univariate_categorical_summary.csv", report)
            self.assertIn(f"etiqueta `{MISSING_IMD_LABEL}`", report)
            self.assertNotIn("`Missing`", report)

    def test_total_clicks_boxplot_summary_uses_stable_result_order(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            summary = eda.total_clicks_by_final_result()

        self.assertEqual(summary["final_result"].astype(str).tolist(), ["Fail", "Pass"])
        self.assertEqual(summary["count"].tolist(), [4, 4])
        self.assertEqual(summary["median"].tolist(), [15.0, 55.0])
        self.assertEqual(summary["iqr"].tolist(), [15.0, 15.0])
        self.assertEqual(summary["lower_whisker"].tolist(), [0, 40])
        self.assertEqual(summary["upper_whisker"].tolist(), [30, 70])

    def test_bivariate_score_boxplots_preserve_missing_score_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(sample_mart(), Path(directory), Path(directory) / "output")
            summary = eda.bivariate_score_boxplots()

        gender = summary[summary["characteristic"] == "gender"]
        self.assertEqual(gender["category"].tolist(), ["F", "M"])
        self.assertEqual(gender["observed_score_count"].tolist(), [3, 2])
        self.assertEqual(gender["missing_score_count"].tolist(), [1, 2])
        self.assertEqual(int(summary["observed_score_count"].sum() / 3), 5)
        self.assertEqual(int(summary["missing_score_count"].sum() / 3), 3)

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

    def test_univariate_profiles_include_only_candidates_with_deterministic_summaries(self):
        frame = sample_mart().assign(academic_risk=[True] * 8)
        with tempfile.TemporaryDirectory() as directory:
            eda = ExploratoryDataAnalysis(frame, Path(directory), Path(directory) / "output")
            numeric = eda.numeric_univariate_profile()
            categorical = eda.categorical_univariate_profile()

        self.assertEqual(numeric["feature"].tolist(), eda.numeric_candidate_features)
        self.assertTrue(set(KEY_COLUMNS + TARGET_COLUMNS + ["academic_risk"]).isdisjoint(numeric["feature"]))
        self.assertTrue(set(KEY_COLUMNS + TARGET_COLUMNS + ["academic_risk"]).isdisjoint(categorical["feature"]))
        self.assertTrue(set(["gender", "age_band", "highest_education"]).isdisjoint(categorical["feature"]))
        self.assertEqual(categorical["feature"].drop_duplicates().tolist(), eda.categorical_candidate_features)
        imd_missing = categorical[(categorical["feature"] == "imd_band") & (categorical["category"] == MISSING_IMD_LABEL)]
        self.assertEqual(imd_missing.iloc[0]["count"], 1)
        self.assertEqual(imd_missing.iloc[0]["share"], 1 / 8)

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
