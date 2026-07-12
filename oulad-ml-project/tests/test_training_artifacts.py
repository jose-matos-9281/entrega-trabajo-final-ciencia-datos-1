import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pandas as pd

from oulad_ml_project.data_generator import write_training_artifacts
from oulad_ml_project.data_sources import KEY_COLUMNS, SQL_PATH, TARGET_COLUMNS, load_training_mart
from oulad_ml_project.preprocessing import DataPreprocessor
from oulad_ml_project.train_ml import ModelTrainer


class TrainingArtifactsTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_NEON_INTEGRATION") == "1",
        "set RUN_NEON_INTEGRATION=1 to query Neon read-only",
    )
    def test_neon_integration_contract(self):
        frame = load_training_mart(30)

        self.assertGreater(len(frame), 0)
        self.assertTrue(set(KEY_COLUMNS + TARGET_COLUMNS).issubset(frame.columns))
        self.assertFalse(any("kongo" in column.lower() for column in frame.columns))

    def test_artifacts_keep_keys_and_separate_targets_without_kongo(self):
        mart = pd.DataFrame(
            {
                "id_student": [2, 1],
                "code_module": ["BBB", "AAA"],
                "code_presentation": ["2014J", "2013J"],
                "total_clicks": [8, 4],
                "has_vle_activity": [1, 1],
                "passed": [1, 0],
                "performance_tier": [2, 0],
                "weighted_assessment_score": [72.5, 35.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_training_artifacts(
                mart,
                Path(directory),
                30,
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            features = pd.read_csv(paths["features"])
            targets = pd.read_csv(paths["targets"])
            full = pd.read_csv(paths["full"])
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        self.assertEqual(list(features.columns[:3]), KEY_COLUMNS)
        self.assertEqual(list(targets.columns), KEY_COLUMNS + TARGET_COLUMNS)
        self.assertTrue(set(TARGET_COLUMNS).isdisjoint(features.columns))
        self.assertEqual(list(full.columns), list(features.columns) + TARGET_COLUMNS)
        self.assertFalse(any("kongo" in column.lower() for column in full.columns))
        self.assertEqual(metadata["cutoff_day"], 30)
        self.assertEqual(metadata["row_count"], 2)
        self.assertIn("no Kongo columns", metadata["full_artifact_compatibility"]["reason"])

    def test_sql_filters_vle_rows_after_cutoff(self):
        connection = duckdb.connect()
        connection.execute("ATTACH ':memory:' AS oulad")
        connection.execute("CREATE SCHEMA oulad.public")
        connection.execute(
            "CREATE TABLE oulad.public.estudiante "
            "(id_estudiante INTEGER, genero VARCHAR, region VARCHAR, nivel_educativo VARCHAR, "
            "imd_band VARCHAR, tiene_discapacidad BOOLEAN, grupo_edad_inicio VARCHAR, grupo_edad_final VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE oulad.public.curso (cd_curso VARCHAR, cd_semestre VARCHAR, duracion_dias INTEGER)"
        )
        connection.execute(
            "CREATE TABLE oulad.public.estudiante_curso "
            "(cd_curso VARCHAR, cd_semestre VARCHAR, id_estudiante INTEGER, fecha_registro_dias INTEGER, "
            "fecha_retiro_dias INTEGER, cant_intentos INTEGER, cant_creditos INTEGER, resultado_final VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE oulad.public.estudiante_recurso "
            "(cd_curso VARCHAR, cd_semestre VARCHAR, id_estudiante INTEGER, id_recurso INTEGER, "
            "fecha_interaccion INTEGER, cant_clicks INTEGER)"
        )
        connection.execute(
            "CREATE TABLE oulad.public.asignaciones "
            "(id_asignacion INTEGER, cd_curso VARCHAR, cd_semestre VARCHAR, tipo_asignacion VARCHAR, "
            "fecha_limite_dias INTEGER, peso_evaluacion DECIMAL(5,2))"
        )
        connection.execute(
            "CREATE TABLE oulad.public.asignacion_estudiante "
            "(id_asignacion INTEGER, id_estudiante INTEGER, dia_entrega INTEGER, is_banked BOOLEAN, puntuacion DECIMAL(5,2))"
        )
        connection.execute("INSERT INTO oulad.public.estudiante VALUES (1, 'M', 'East', 'Bachelor', NULL, false, '0-35', '0-35'), (2, 'F', 'North', 'Master', NULL, false, '35-55', '35-55'), (3, 'F', 'South', 'No Formal', NULL, false, '55-', '55-')")
        connection.execute("INSERT INTO oulad.public.curso VALUES ('AAA', '2013J', 100)")
        connection.execute("INSERT INTO oulad.public.estudiante_curso VALUES ('AAA', '2013J', 1, -10, NULL, 0, 60, 'Pass'), ('AAA', '2013J', 2, 31, NULL, 0, 60, 'Pass'), ('AAA', '2013J', 3, NULL, NULL, 0, 60, 'Pass')")
        connection.execute("INSERT INTO oulad.public.estudiante_recurso VALUES ('AAA', '2013J', 1, 10, 5, 4), ('AAA', '2013J', 1, 11, 31, 99)")
        connection.execute("INSERT INTO oulad.public.asignaciones VALUES (1, 'AAA', '2013J', 'TMA', 40, 100)")
        connection.execute("INSERT INTO oulad.public.asignacion_estudiante VALUES (1, 1, 40, false, 70)")

        frame = connection.execute(SQL_PATH.read_text(encoding="utf-8"), [30]).fetchdf()
        connection.close()

        self.assertEqual(frame.loc[0, "total_clicks"], 4)
        self.assertEqual(frame.loc[0, "active_days"], 1)
        self.assertEqual(frame.loc[0, "vle_events"], 1)
        self.assertEqual(frame.loc[0, "vle_sites"], 1)
        self.assertEqual(frame.loc[0, "has_vle_activity"], 1)
        self.assertEqual(frame.loc[0, "passed"], 1)
        self.assertNotIn("resultado_final", frame.columns)
        self.assertEqual(frame["id_student"].tolist(), [1])

    def test_load_training_mart_loads_postgres_without_installing(self):
        connection = Mock()
        query_result = Mock(fetchdf=Mock(return_value=pd.DataFrame({
            "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013J"]
        })))
        connection.execute.side_effect = [None, None, query_result]

        with patch("oulad_ml_project.data_sources._postgres_dsn", return_value="postgresql://redacted"), \
             patch("oulad_ml_project.data_sources.duckdb.connect", return_value=connection):
            load_training_mart(30)

        executed_sql = [call.args[0] for call in connection.execute.call_args_list]
        self.assertEqual(executed_sql[0], "LOAD postgres")
        self.assertFalse(any(statement == "INSTALL postgres" for statement in executed_sql))
        self.assertIn("READ_ONLY", executed_sql[1])

    def test_load_training_mart_explains_unavailable_postgres_extension(self):
        connection = Mock()
        connection.execute.side_effect = duckdb.Error("extension unavailable")

        with patch("oulad_ml_project.data_sources._postgres_dsn", return_value="postgresql://redacted"), \
             patch("oulad_ml_project.data_sources.duckdb.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "installed and loadable"):
                load_training_mart(30)

        self.assertEqual(connection.execute.call_args.args[0], "LOAD postgres")
        connection.close.assert_called_once()

    def test_preprocessor_uses_ordinal_mapping_for_highest_education(self):
        frame = pd.DataFrame({"highest_education": ["Bachelor", "No Formal", "Master"]})

        processed = DataPreprocessor(frame).encode_ordinal()

        self.assertEqual(processed["highest_education_encoded"].tolist(), [3, 0, 4])
        self.assertNotIn("highest_education_enc", processed.columns)

    def test_preprocessor_does_not_impute_missing_targets(self):
        frame = pd.DataFrame(
            {
                "id_student": [1, 2],
                "total_clicks": [None, 4],
                "passed": [1, 0],
                "performance_tier": [2, 0],
                "weighted_assessment_score": [None, 80.0],
            }
        )

        processed = DataPreprocessor(frame).handle_missing()

        self.assertEqual(processed.loc[0, "total_clicks"], 4)
        self.assertTrue(pd.isna(processed.loc[0, "weighted_assessment_score"]))

    def test_grouped_splits_keep_students_disjoint_and_exclude_identity_features(self):
        frame = pd.DataFrame(
            {
                "id_student": [1, 1, 2, 2, 3, 3, 4, 4],
                "total_clicks": [1, 2, 3, 4, 5, 6, 7, 8],
                "passed": [0, 0, 1, 1, 0, 0, 1, 1],
                "performance_tier": [0, 0, 1, 1, 2, 2, 3, 3],
                "weighted_assessment_score": [None, 40.0, 60.0, None, 70.0, 75.0, 80.0, 85.0],
            }
        )
        preprocessor = DataPreprocessor(frame)
        X, targets, features = preprocessor.prepare_features(TARGET_COLUMNS)

        self.assertNotIn("id_student", features)
        self.assertNotIn("id_student", X.columns)
        self.assertTrue(set(TARGET_COLUMNS).isdisjoint(X.columns))
        self.assertTrue(all(target.index.equals(X.index) for target in targets.values()))

        trainer = ModelTrainer(X, targets, Path("."), frame["id_student"])
        binary_split = trainer.split_data("passed")
        ordinal_split = trainer.split_data("performance_tier")
        scored_rows = targets["weighted_assessment_score"].notna()
        regression_split = trainer.split_grouped(
            X.loc[scored_rows], targets["weighted_assessment_score"].loc[scored_rows]
        )

        for X_train, X_test, *_ in (binary_split, ordinal_split, regression_split):
            train_students = set(frame.loc[X_train.index, "id_student"])
            test_students = set(frame.loc[X_test.index, "id_student"])
            self.assertFalse(train_students & test_students)


if __name__ == "__main__":
    unittest.main()
