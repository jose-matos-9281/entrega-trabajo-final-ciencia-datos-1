"""Leakage-safe exploratory data analysis for the OULAD training artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_sources import KEY_COLUMNS, TARGET_COLUMNS


SCORE_COLUMN = "weighted_assessment_score"
PASS_COLUMN = "passed"
ENGAGEMENT_COLUMN = "total_clicks"
SPARSE_GROUP_MINIMUM = 30
MISSING_IMD_LABEL = "Sin dato"

DISPLAY_LABELS = {
    "code_module": "Código de curso",
    "code_presentation": "Código de presentación",
    "gender": "Género",
    "age_band": "Grupo etario",
    "highest_education": "Nivel educativo más alto",
    "imd_band": "Banda IMD",
    "total_clicks": "Total de clics",
    "weighted_assessment_score": "Puntuación ponderada de evaluación",
}


class ExploratoryDataAnalysis:
    """Create deterministic, descriptive EDA artifacts for an enrollment mart.

    The constructor remains compatible with the pipeline's existing
    ``(df, data_dir, output_dir)`` interface. Analyses never use target columns
    as candidate features.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, output_dir: Path):
        self.df = df.copy()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            column
            for column in self.df.columns
            if pd.api.types.is_object_dtype(self.df[column])
            or pd.api.types.is_string_dtype(self.df[column])
            or isinstance(self.df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(self.df[column])
        ]
        self.generated_files: list[Path] = []

    @property
    def candidate_features(self) -> list[str]:
        """Return model candidates, excluding enrollment identifiers and outcomes."""
        return [
            column
            for column in self.df.columns
            if column not in KEY_COLUMNS and column not in TARGET_COLUMNS
        ]

    @property
    def numeric_candidate_features(self) -> list[str]:
        return [column for column in self.candidate_features if pd.api.types.is_numeric_dtype(self.df[column])]

    def _write_csv(self, frame: pd.DataFrame, filename: str, *, index: bool = False) -> Path:
        path = self.output_dir / filename
        frame.to_csv(path, index=index, float_format="%.6f")
        self.generated_files.append(path)
        return path

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        path = self.fig_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        self.generated_files.append(path)
        return path

    @staticmethod
    def _display_label(column: str) -> str:
        return DISPLAY_LABELS.get(column, column.replace("_", " ").capitalize())

    def validate_contract(self) -> None:
        """Reject artifacts that cannot represent one row per enrollment."""
        required_columns = set(KEY_COLUMNS + TARGET_COLUMNS)
        missing_columns = sorted(required_columns.difference(self.df.columns))
        if missing_columns:
            raise ValueError(f"La entrada del EDA no contiene las columnas requeridas: {missing_columns}")
        if self.df.duplicated(KEY_COLUMNS).any():
            raise ValueError("La entrada del EDA debe tener una fila por clave de matrícula")

    def cohort_snapshot(self) -> pd.DataFrame:
        rows = [
            ("enrollment_rows", len(self.df)),
            ("columns", len(self.df.columns)),
            ("unique_students", self.df["id_student"].nunique()),
            ("courses", self.df["code_module"].nunique()),
            ("presentations", self.df["code_presentation"].nunique()),
            ("candidate_features", len(self.candidate_features)),
        ]
        snapshot = pd.DataFrame(rows, columns=["metric", "value"])
        self._write_csv(snapshot, "cohort_snapshot.csv")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(self.df[ENGAGEMENT_COLUMN], bins=40, color="#2a6f97", edgecolor="white")
        axis.set_title("Interacción de la cohorte: total de clics antes del corte")
        axis.set_xlabel(self._display_label(ENGAGEMENT_COLUMN))
        axis.set_ylabel("Filas de matrícula")
        self._save_figure(figure, "cohort_engagement_total_clicks.png")
        return snapshot

    def data_quality(self) -> pd.DataFrame:
        checks = pd.DataFrame(
            [
                ("required_columns_present", True, 0),
                ("duplicate_enrollment_keys", False, int(self.df.duplicated(KEY_COLUMNS).sum())),
                ("zero_vle_engagement", False, int((self.df[ENGAGEMENT_COLUMN] == 0).sum())),
            ],
            columns=["check", "failed", "count"],
        )
        self._write_csv(checks, "data_quality_checks.csv")
        return checks

    def categorical_profile(self) -> pd.DataFrame:
        profile_columns = [
            column
            for column in ["code_module", "code_presentation", "gender", "age_band", "imd_band"]
            if column in self.df.columns
        ]
        rows = []
        for column in profile_columns:
            values = self.df[column].fillna(MISSING_IMD_LABEL) if column == "imd_band" else self.df[column]
            counts = values.value_counts(dropna=False).sort_index()
            rows.extend((column, str(level), int(count), count / len(self.df)) for level, count in counts.items())
        profile = pd.DataFrame(rows, columns=["column", "category", "count", "share"])
        self._write_csv(profile, "categorical_profile.csv")

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, column in zip(axes, ["code_module", "imd_band"]):
            values = self.df[column].fillna(MISSING_IMD_LABEL) if column == "imd_band" else self.df[column]
            values.value_counts().sort_index().plot.bar(ax=axis, color="#4c956c")
            axis.set_title(f"Perfil de la cohorte: {self._display_label(column)}")
            axis.set_xlabel("")
            axis.set_ylabel("Filas de matrícula")
            axis.tick_params(axis="x", rotation=45)
        self._save_figure(figure, "categorical_cohort_profile.png")
        return profile

    def missingness_and_scores(self) -> pd.DataFrame:
        missing = self.df.isna().sum().rename("missing_count").to_frame()
        missing["missing_share"] = missing["missing_count"] / len(self.df)
        missing["available_count"] = len(self.df) - missing["missing_count"]
        missing.index.name = "column"
        missing = missing.reset_index().sort_values(["missing_count", "column"], ascending=[False, True])
        self._write_csv(missing, "missingness_and_score_availability.csv")

        figure, axis = plt.subplots(figsize=(9, 4.5))
        displayed = missing[missing["missing_count"] > 0]
        axis.bar(displayed["column"], displayed["missing_share"] * 100, color="#bc4749")
        axis.set_title("Valores faltantes y disponibilidad de puntuaciones observadas")
        axis.set_xlabel("Variable")
        axis.set_ylabel("Filas de matrícula con valores faltantes (%)")
        axis.set_xticks(range(len(displayed)), [self._display_label(column) for column in displayed["column"]])
        axis.tick_params(axis="x", rotation=30)
        self._save_figure(figure, "missingness_and_observed_scores.png")
        return missing

    def candidate_spearman_correlations(self) -> pd.DataFrame:
        features = self.numeric_candidate_features
        correlation = self.df[features].corr(method="spearman") if features else pd.DataFrame()
        correlation.index.name = "feature"
        self._write_csv(correlation.reset_index(), "candidate_feature_spearman_correlations.csv")

        figure, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_title("Correlaciones de Spearman entre variables candidatas")
        display_features = [self._display_label(feature) for feature in features]
        axis.set_xticks(range(len(features)), display_features, rotation=60, ha="right")
        axis.set_yticks(range(len(features)), display_features)
        figure.colorbar(image, ax=axis, label="Correlación de Spearman")
        self._save_figure(figure, "candidate_feature_spearman_correlations.png")
        return correlation

    def collinearity_summary(self, correlation: pd.DataFrame) -> pd.DataFrame:
        pairs = []
        for position, left in enumerate(correlation.columns):
            for right in correlation.columns[position + 1 :]:
                value = correlation.loc[left, right]
                if pd.notna(value):
                    pairs.append((left, right, value, abs(value), abs(value) >= 0.95))
        summary = pd.DataFrame(
            pairs,
            columns=["feature_a", "feature_b", "spearman_correlation", "absolute_correlation", "near_duplicate"],
        ).sort_values("absolute_correlation", ascending=False, kind="stable")
        self._write_csv(summary, "collinearity_and_near_duplicate_summary.csv")
        return summary

    def _engagement_quantiles(self) -> pd.Series:
        ranks = self.df[ENGAGEMENT_COLUMN].rank(method="first")
        return pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    def outcome_relationships(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        working = self.df.assign(engagement_quantile=self._engagement_quantiles())
        engagement = working.groupby("engagement_quantile", observed=True).agg(
            n_total=(PASS_COLUMN, "size"),
            pass_rate=(PASS_COLUMN, "mean"),
            n_scored=(SCORE_COLUMN, "count"),
            observed_score_mean=(SCORE_COLUMN, "mean"),
        ).reset_index()
        engagement["observed_score_share"] = engagement["n_scored"] / engagement["n_total"]
        self._write_csv(engagement, "engagement_outcome_summary.csv")

        characteristics = []
        for column in ["gender", "age_band", "highest_education", "imd_band"]:
            if column not in working:
                continue
            values = working[column].fillna(MISSING_IMD_LABEL) if column == "imd_band" else working[column]
            grouped = working.assign(**{column: values}).groupby(column, observed=True).agg(
                n_total=(PASS_COLUMN, "size"), pass_rate=(PASS_COLUMN, "mean")
            ).reset_index().rename(columns={column: "category"})
            grouped.insert(0, "characteristic", column)
            characteristics.append(grouped)
        characteristic_summary = pd.concat(characteristics, ignore_index=True)
        self._write_csv(characteristic_summary, "pass_rate_by_student_characteristics.csv")

        course = working.groupby(["code_module", "code_presentation"], observed=True).agg(
            n_total=(PASS_COLUMN, "size"),
            pass_rate=(PASS_COLUMN, "mean"),
            n_scored=(SCORE_COLUMN, "count"),
            observed_score_mean=(SCORE_COLUMN, "mean"),
        ).reset_index()
        course["suppressed"] = course["n_total"] < SPARSE_GROUP_MINIMUM
        course.loc[course["suppressed"], ["pass_rate", "observed_score_mean"]] = np.nan
        course["observed_score_share"] = course["n_scored"] / course["n_total"]
        self._write_csv(course, "course_presentation_outcome_summary.csv")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(engagement["engagement_quantile"].astype(str), engagement["pass_rate"] * 100, color="#386641")
        axis.set_title("Tasa de aprobación por cuartil de interacción antes del corte")
        axis.set_xlabel("Cuartil de interacción según total de clics")
        axis.set_ylabel("Tasa de aprobación (%)")
        self._save_figure(figure, "pass_rate_by_engagement_quantile.png")

        imd = characteristic_summary[characteristic_summary["characteristic"] == "imd_band"]
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.bar(imd["category"], imd["pass_rate"] * 100, color="#6a4c93")
        axis.set_title("Tasa de aprobación por banda IMD, incluidos los valores faltantes")
        axis.set_xlabel(self._display_label("imd_band"))
        axis.set_ylabel("Tasa de aprobación (%)")
        axis.tick_params(axis="x", rotation=45)
        self._save_figure(figure, "pass_rate_by_student_characteristics.png")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        labels = [f"{row.engagement_quantile}\n{row.n_scored}/{row.n_total} con puntuación" for row in engagement.itertuples()]
        axis.bar(labels, engagement["observed_score_mean"], color="#f4a261")
        axis.set_title("Puntuación observada de evaluación por interacción antes del corte")
        axis.set_xlabel("Cuartil de interacción (con puntuación/total)")
        axis.set_ylabel("Puntuación media observada de evaluación")
        self._save_figure(figure, "observed_score_by_engagement_quantile.png")

        visible_course = course[~course["suppressed"]].copy()
        figure, axis = plt.subplots(figsize=(10, 4.5))
        labels = visible_course["code_module"] + " " + visible_course["code_presentation"]
        axis.bar(labels, visible_course["pass_rate"] * 100, color="#0077b6")
        axis.set_title(f"Tasas de aprobación por curso y presentación (grupos n >= {SPARSE_GROUP_MINIMUM})")
        axis.set_xlabel("Curso y presentación")
        axis.set_ylabel("Tasa de aprobación (%)")
        axis.tick_params(axis="x", rotation=45)
        self._save_figure(figure, "course_presentation_outcomes.png")
        return engagement, characteristic_summary, course

    def write_report(
        self,
        snapshot: pd.DataFrame,
        quality: pd.DataFrame,
        missingness: pd.DataFrame,
        collinearity: pd.DataFrame,
        engagement: pd.DataFrame,
        course: pd.DataFrame,
    ) -> Path:
        values = snapshot.set_index("metric")["value"]
        score = missingness.set_index("column").loc[SCORE_COLUMN]
        imd = missingness.set_index("column").loc["imd_band"]
        zero_vle = int(quality.loc[quality["check"] == "zero_vle_engagement", "count"].iloc[0])
        near_duplicates = int(collinearity["near_duplicate"].sum()) if not collinearity.empty else 0
        near_duplicate_label = "par" if near_duplicates == 1 else "pares"
        first_quantile = engagement.iloc[0]
        last_quantile = engagement.iloc[-1]
        suppressed_groups = int(course["suppressed"].sum())
        #figure_paths = [path for path in self.generated_files if path.parent == self.fig_dir]

        lines = [
            "# Análisis exploratorio de datos de OULAD con prevención de fuga de información",
            "",
            "## Contrato del conjunto de datos",
            "",
            f"El artefacto contiene {int(values['enrollment_rows']):,} filas de matrícula y {int(values['columns'])} columnas. Representa a {int(values['unique_students']):,} estudiantes en {int(values['courses'])} cursos y {int(values['presentations'])} presentaciones; la clave de matrícula es `{', '.join(KEY_COLUMNS)}`.",
            "",
            f"El EDA considera las {int(values['candidate_features'])} columnas que no son claves ni resultados como variables candidatas. `{', '.join(TARGET_COLUMNS)}` se usan únicamente como resultados y se excluyen de las correlaciones entre variables candidatas y de las comprobaciones de colinealidad.",
            "",
            "## Calidad de los datos y disponibilidad de etiquetas",
            "",
            f"Están presentes todas las columnas de claves y resultados requeridas, sin claves de matrícula duplicadas. {zero_vle:,} filas de matrícula tienen cero clics en el VLE antes del corte temporal.",
            "",
            f"`{SCORE_COLUMN}` se observa en {int(score['available_count']):,}/{len(self.df):,} filas de matrícula ({score['available_count'] / len(self.df):.2%}) y falta en {int(score['missing_count']):,}/{len(self.df):,} ({score['missing_share']:.2%}). `imd_band` falta en {int(imd['missing_count']):,}/{len(self.df):,} filas ({imd['missing_share']:.2%}) y se muestra con la etiqueta `{MISSING_IMD_LABEL}` en las vistas de características.",
            "",
            "![Valores faltantes y puntuaciones observadas](figures/missingness_and_observed_scores.png)",
            "",
            "## Interacción de la cohorte",
            "",
            f"La distribución de interacción resume `total_clicks`, una medida agregada de uso del VLE antes del corte temporal, para cada fila de matrícula. Tras una segmentación determinista basada en rangos, cada cuartil de interacción contiene entre {int(first_quantile['n_total']):,} y {int(last_quantile['n_total']):,} filas de matrícula.",
            "",
            "![Interacción de la cohorte](figures/cohort_engagement_total_clicks.png)",
            "",
            "![Perfil categórico de la cohorte](figures/categorical_cohort_profile.png)",
            "",
            "## Relaciones con los resultados",
            "",
            f"Las tasas de aprobación observadas van de {first_quantile['pass_rate']:.2%} en el cuartil de interacción Q1 a {last_quantile['pass_rate']:.2%} en Q4. Estas comparaciones descriptivas no establecen una relación causal ni un resultado de un modelo predictivo.",
            "",
            f"La comparación de puntuaciones observadas informa tanto la disponibilidad como el denominador: Q1 tiene {int(first_quantile['n_scored']):,}/{int(first_quantile['n_total']):,} filas con puntuación y Q4 tiene {int(last_quantile['n_scored']):,}/{int(last_quantile['n_total']):,}. Los resultados por curso y presentación suprimen las tasas y las medias de puntuación observada de {suppressed_groups} grupos con n inferior a {SPARSE_GROUP_MINIMUM}.",
            "",
            "![Tasa de aprobación por interacción](figures/pass_rate_by_engagement_quantile.png)",
            "",
            "![Tasa de aprobación por características del estudiantado](figures/pass_rate_by_student_characteristics.png)",
            "",
            "![Puntuación observada por interacción](figures/observed_score_by_engagement_quantile.png)",
            "",
            "![Resultados por curso y presentación](figures/course_presentation_outcomes.png)",
            "",
            "## Preparación para el modelado y exclusiones por fuga de información",
            "",
            f"Las correlaciones de Spearman usan únicamente variables candidatas numéricas y excluyen los identificadores de matrícula y todos los resultados. El resumen de colinealidad identifica {near_duplicates} {near_duplicate_label} de variables candidatas con correlación absoluta de Spearman de al menos 0.95; es un artefacto de cribado, no de selección de variables.",
            "",
            "No se genera aquí una matriz de confusión, porque corresponde a la evaluación del modelo con datos de prueba reservados y no al análisis exploratorio.",
            "",
            "![Correlaciones entre variables candidatas](figures/candidate_feature_spearman_correlations.png)",
            "",
            "## Apéndice: archivos generados",
            "",
        ]
        # lines.extend(f"- `{path.relative_to(self.output_dir).as_posix()}`" for path in sorted(self.generated_files))
        # lines.append("- `eda_report.md`")
        report = self.output_dir / "eda_report.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.generated_files.append(report)
        return report

    def run_all(self) -> list[Path]:
        """Run all leakage-safe descriptive analyses and return their manifest."""
        self.validate_contract()
        snapshot = self.cohort_snapshot()
        quality = self.data_quality()
        self.categorical_profile()
        missingness = self.missingness_and_scores()
        correlation = self.candidate_spearman_correlations()
        collinearity = self.collinearity_summary(correlation)
        engagement, _, course = self.outcome_relationships()
        self.write_report(snapshot, quality, missingness, collinearity, engagement, course)
        return self.generated_files


def main() -> None:
    """Run the full exploratory data analysis from a CSV artifact."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Ejecuta el análisis exploratorio de datos de OULAD con prevención de fuga de información.")
    parser.add_argument("--input-csv", type=Path, default=project_dir / "data" / "oulad_training_full.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "output" / "eda")
    args = parser.parse_args()
    manifest = ExploratoryDataAnalysis(pd.read_csv(args.input_csv), args.input_csv.parent, args.output_dir).run_all()
    print(f"Se generaron {len(manifest)} artefactos de EDA en {args.output_dir}")


if __name__ == "__main__":
    main()
