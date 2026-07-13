"""Leakage-safe exploratory data analysis for the OULAD training artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_sources import KEY_COLUMNS, TARGET_COLUMNS


SCORE_COLUMN = "weighted_assessment_score"
ENGAGEMENT_COLUMN = "total_clicks"
DERIVED_OUTCOME_COLUMNS = ("academic_risk",)
DESCRIPTIVE_ONLY_COLUMNS = ("gender", "age_band", "highest_education")
NUMERIC_IDENTIFIER_PREFIXES = ("id_",)
SPARSE_GROUP_MINIMUM = 30
MISSING_IMD_LABEL = "Sin dato"
RISK_RESULTS = {"Withdrawn", "Fail"}
NON_RISK_RESULTS = {"Pass", "Distinction"}
FINAL_RESULT_ORDER = ["Withdrawn", "Fail", "Pass", "Distinction"]
BIVARIATE_CATEGORY_ORDERS = {
    "gender": ["F", "M"],
    "age_band": ["0-35", "35-55", "55<="],
    "highest_education": [
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification",
    ],
}
RISK_LABEL = "Riesgo académico"
NON_RISK_LABEL = "Resultado favorable"

DISPLAY_LABELS = {
    "code_module": "Código de curso",
    "code_presentation": "Código de presentación",
    "gender": "Género",
    "age_band": "Grupo etario",
    "highest_education": "Nivel educativo más alto",
    "imd_band": "Banda IMD",
    "num_of_prev_attempts": "Intentos previos",
    "studied_credits": "Créditos cursados",
    "date_registration": "Días de registro",
    "course_duration_days": "Duración del curso (días)",
    "total_clicks": "Total de clics",
    "active_days": "Días activos",
    "vle_events": "Eventos VLE",
    "vle_sites": "Sitios VLE",
    "has_vle_activity": "Tiene actividad VLE",
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
        metadata_path = self.data_dir / "oulad_training_metadata.json"
        self.cutoff_day = json.loads(metadata_path.read_text(encoding="utf-8")).get("cutoff_day") if metadata_path.exists() else None

    @property
    def candidate_features(self) -> list[str]:
        """Return model candidates, excluding enrollment identifiers and outcomes."""
        return [
            column
            for column in self.df.columns
            if column not in KEY_COLUMNS + TARGET_COLUMNS + list(DERIVED_OUTCOME_COLUMNS + DESCRIPTIVE_ONLY_COLUMNS)
        ]

    @property
    def numeric_candidate_features(self) -> list[str]:
        return [
            column
            for column in self.candidate_features
            if pd.api.types.is_numeric_dtype(self.df[column])
            and not pd.api.types.is_bool_dtype(self.df[column])
            and not column.startswith(NUMERIC_IDENTIFIER_PREFIXES)
        ]

    @property
    def categorical_candidate_features(self) -> list[str]:
        return [column for column in self.candidate_features if column in self.categorical_cols]

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
        known_results = RISK_RESULTS | NON_RISK_RESULTS
        unexpected_results = sorted(set(self.df["final_result"].dropna()).difference(known_results))
        if self.df["final_result"].isna().any() or unexpected_results:
            raise ValueError("final_result debe contener únicamente Withdrawn, Fail, Pass o Distinction")

    def _remove_obsolete_generated_artifacts(self) -> None:
        """Remove only former EDA outputs whose names no longer describe the hypothesis."""
        obsolete = [
            self.output_dir / "course_presentation_outcome_summary.csv",
            self.output_dir / "engagement_outcome_summary.csv",
            self.output_dir / "pass_rate_by_student_characteristics.csv",
            self.fig_dir / "course_presentation_outcomes.png",
            self.fig_dir / "observed_score_by_engagement_quantile.png",
            self.fig_dir / "pass_rate_by_engagement_quantile.png",
            self.fig_dir / "pass_rate_by_student_characteristics.png",
        ]
        for path in obsolete:
            path.unlink(missing_ok=True)

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

    def numeric_univariate_profile(self) -> pd.DataFrame:
        """Summarize and plot numeric candidate features without outcomes or identifiers."""
        features = self.numeric_candidate_features
        rows = []
        for column in features:
            values = self.df[column]
            rows.append(
                (
                    column,
                    int(values.count()),
                    int(values.isna().sum()),
                    values.mean(),
                    values.median(),
                    values.std(),
                    values.min(),
                    values.quantile(0.25),
                    values.quantile(0.75),
                    values.max(),
                )
            )
        summary = pd.DataFrame(
            rows,
            columns=["feature", "count", "missing_count", "mean", "median", "std", "min", "q1", "q3", "max"],
        )
        self._write_csv(summary, "univariate_numeric_summary.csv")

        if features:
            columns = 3
            rows_count = int(np.ceil(len(features) / columns))
            figure, axes = plt.subplots(rows_count, columns, figsize=(14, 3.8 * rows_count), squeeze=False)
            for axis, column in zip(axes.flat, features):
                values = self.df[column].dropna()
                bin_count = min(40, max(5, int(np.sqrt(len(values))))) if len(values) else 5
                axis.hist(values, bins=bin_count, color="#2a6f97", edgecolor="white")
                zero_count = int((values == 0).sum())
                axis.set_title(self._display_label(column))
                axis.set_xlabel("")
                axis.set_ylabel("Filas de matrícula")
                if zero_count:
                    axis.text(0.98, 0.95, f"Ceros: {zero_count:,}", transform=axis.transAxes, ha="right", va="top")
            for axis in axes.flat[len(features) :]:
                axis.remove()
            self._save_figure(figure, "univariate_numeric_histograms.png")
        return summary

    def categorical_univariate_profile(self) -> pd.DataFrame:
        """Summarize and plot categorical candidate features, retaining missing values."""
        features = self.categorical_candidate_features
        rows = []
        for column in features:
            values = self.df[column].fillna(MISSING_IMD_LABEL).astype(str)
            counts = values.value_counts(dropna=False).sort_index(kind="stable")
            rows.extend((column, category, int(count), count / len(self.df)) for category, count in counts.items())
        summary = pd.DataFrame(rows, columns=["feature", "category", "count", "share"])
        self._write_csv(summary, "univariate_categorical_summary.csv")

        if features:
            columns = 2
            rows_count = int(np.ceil(len(features) / columns))
            figure, axes = plt.subplots(rows_count, columns, figsize=(14, 4.4 * rows_count), squeeze=False)
            for axis, column in zip(axes.flat, features):
                counts = self.df[column].fillna(MISSING_IMD_LABEL).astype(str).value_counts(dropna=False).sort_index(kind="stable")
                if max(map(len, counts.index)) > 16:
                    axis.barh(counts.index, counts.values, color="#4c956c")
                    axis.set_xlabel("Filas de matrícula")
                    axis.set_ylabel("")
                else:
                    axis.bar(counts.index, counts.values, color="#4c956c")
                    axis.set_xlabel("")
                    axis.set_ylabel("Filas de matrícula")
                    axis.tick_params(axis="x", rotation=35)
                axis.set_title(self._display_label(column))
            for axis in axes.flat[len(features) :]:
                axis.remove()
            self._save_figure(figure, "univariate_categorical_frequencies.png")
        return summary

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

    def total_clicks_by_final_result(self) -> pd.DataFrame:
        """Describe early engagement distributions by observed final result."""
        working = self.df[["final_result", ENGAGEMENT_COLUMN]].copy()
        working["final_result"] = pd.Categorical(
            working["final_result"], categories=FINAL_RESULT_ORDER, ordered=True
        )
        summary = working.groupby("final_result", observed=True)[ENGAGEMENT_COLUMN].agg(
            count="count",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            min="min",
            max="max",
        ).reset_index()
        summary["iqr"] = summary["q3"] - summary["q1"]
        summary["lower_whisker"] = summary.apply(
            lambda row: working.loc[
                (working["final_result"] == row["final_result"])
                & (working[ENGAGEMENT_COLUMN] >= row["q1"] - 1.5 * row["iqr"]),
                ENGAGEMENT_COLUMN,
            ].min(),
            axis=1,
        )
        summary["upper_whisker"] = summary.apply(
            lambda row: working.loc[
                (working["final_result"] == row["final_result"])
                & (working[ENGAGEMENT_COLUMN] <= row["q3"] + 1.5 * row["iqr"]),
                ENGAGEMENT_COLUMN,
            ].max(),
            axis=1,
        )
        summary = summary[
            ["final_result", "count", "median", "q1", "q3", "iqr", "min", "max", "lower_whisker", "upper_whisker"]
        ]
        self._write_csv(summary, "total_clicks_by_final_result_boxplot_summary.csv")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        values_by_result = [
            working.loc[working["final_result"] == result, ENGAGEMENT_COLUMN].to_numpy()
            for result in summary["final_result"]
        ]
        axis.boxplot(values_by_result, tick_labels=summary["final_result"].astype(str))
        axis.set_title("Distribución de clics antes del corte por resultado académico final")
        axis.set_xlabel("Resultado final")
        axis.set_ylabel("Total de clics antes del corte")
        self._save_figure(figure, "total_clicks_by_final_result_boxplot.png")
        return summary

    @staticmethod
    def _ordered_categories(column: str, values: pd.Series) -> list[str]:
        observed = set(values.astype(str))
        preferred = BIVARIATE_CATEGORY_ORDERS[column]
        return [category for category in preferred if category in observed] + sorted(observed.difference(preferred))

    def bivariate_score_boxplots(self) -> pd.DataFrame:
        """Describe observed score distributions by sensitive cohort characteristics."""
        characteristics = list(DESCRIPTIVE_ONLY_COLUMNS)
        rows = []
        figure, axes = plt.subplots(1, len(characteristics), figsize=(16, 5.5), squeeze=False)

        for axis, column in zip(axes.flat, characteristics):
            working = self.df[[column, SCORE_COLUMN]].copy()
            working[column] = working[column].fillna(MISSING_IMD_LABEL).astype(str)
            categories = self._ordered_categories(column, working[column])
            values_by_category = []
            for category in categories:
                scores = working.loc[working[column] == category, SCORE_COLUMN]
                observed_scores = scores.dropna()
                q1 = observed_scores.quantile(0.25)
                q3 = observed_scores.quantile(0.75)
                iqr = q3 - q1
                rows.append(
                    (
                        column,
                        category,
                        int(observed_scores.count()),
                        int(scores.isna().sum()),
                        observed_scores.median(),
                        q1,
                        q3,
                        iqr,
                        observed_scores.min(),
                        observed_scores.max(),
                        observed_scores[observed_scores >= q1 - 1.5 * iqr].min(),
                        observed_scores[observed_scores <= q3 + 1.5 * iqr].max(),
                    )
                )
                values_by_category.append(observed_scores.to_numpy())

            axis.boxplot(values_by_category, tick_labels=[f"{category}\nn={len(values):,}" for category, values in zip(categories, values_by_category)])
            axis.set_title(f"{self._display_label(column)}")
            axis.set_xlabel("")
            axis.set_ylabel(self._display_label(SCORE_COLUMN))
            axis.tick_params(axis="x", rotation=30)

        summary = pd.DataFrame(
            rows,
            columns=[
                "characteristic", "category", "observed_score_count", "missing_score_count",
                "median", "q1", "q3", "iqr", "min", "max", "lower_whisker", "upper_whisker",
            ],
        )
        self._write_csv(summary, "bivariate_score_boxplots_summary.csv")
        self._save_figure(figure, "bivariate_score_boxplots.png")
        return summary

    def academic_risk_analysis(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Describe final outcomes without adding outcomes to model candidate features."""
        working = self.df.assign(
            engagement_quantile=self._engagement_quantiles(),
            academic_risk=self.df["final_result"].isin(RISK_RESULTS),
        )
        final_result = working.groupby("final_result", observed=True).size().rename("n_total").reset_index()
        final_result["share"] = final_result["n_total"] / len(working)
        self._write_csv(final_result, "final_result_distribution.csv")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(final_result["final_result"], final_result["n_total"], color="#386641")
        axis.set_title("Distribución del resultado académico final")
        axis.set_xlabel("Resultado final")
        axis.set_ylabel("Filas de matrícula")
        self._save_figure(figure, "final_result_distribution.png")

        risk = working.groupby("academic_risk", observed=True).size().rename("n_total").reset_index()
        risk["academic_risk"] = np.where(risk["academic_risk"], RISK_LABEL, NON_RISK_LABEL)
        risk["share"] = risk["n_total"] / len(working)
        self._write_csv(risk, "academic_risk_distribution.csv")

        engagement = working.groupby("engagement_quantile", observed=True).agg(
            n_total=("academic_risk", "size"),
            n_academic_risk=("academic_risk", "sum"),
        ).reset_index()
        engagement["academic_risk_rate"] = engagement["n_academic_risk"] / engagement["n_total"]
        self._write_csv(engagement, "academic_risk_by_engagement_quartile.csv")

        course = working.groupby(["code_module", "code_presentation"], observed=True).agg(
            n_total=("academic_risk", "size"),
            n_academic_risk=("academic_risk", "sum"),
        ).reset_index()
        course["academic_risk_rate"] = course["n_academic_risk"] / course["n_total"]
        course["suppressed"] = course["n_total"] < SPARSE_GROUP_MINIMUM
        course.loc[course["suppressed"], "academic_risk_rate"] = np.nan
        self._write_csv(course, "academic_risk_by_course_presentation.csv")

        score_missingness = working.groupby("final_result", observed=True).agg(
            n_total=(SCORE_COLUMN, "size"),
            n_score_missing=(SCORE_COLUMN, lambda values: values.isna().sum()),
        ).reset_index()
        score_missingness["score_missing_rate"] = score_missingness["n_score_missing"] / score_missingness["n_total"]
        self._write_csv(score_missingness, "score_missingness_by_final_result.csv")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(score_missingness["final_result"], score_missingness["score_missing_rate"] * 100, color="#f4a261")
        axis.set_title("Faltantes de puntuación por resultado académico final")
        axis.set_xlabel("Resultado final")
        axis.set_ylabel("Filas sin puntuación ponderada (%)")
        self._save_figure(figure, "score_missingness_by_final_result.png")

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(engagement["engagement_quantile"].astype(str), engagement["academic_risk_rate"] * 100, color="#bc4749")
        axis.set_title("Tasa de riesgo académico por cuartil de interacción antes del corte")
        axis.set_xlabel("Cuartil de interacción según total de clics")
        axis.set_ylabel("Tasa de riesgo académico (%)")
        self._save_figure(figure, "academic_risk_by_engagement_quartile.png")

        visible_course = course[~course["suppressed"]].copy()
        figure, axis = plt.subplots(figsize=(10, 4.5))
        labels = visible_course["code_module"] + " " + visible_course["code_presentation"]
        axis.bar(labels, visible_course["academic_risk_rate"] * 100, color="#0077b6")
        axis.set_title(f"Tasas de riesgo académico por curso y presentación (grupos n >= {SPARSE_GROUP_MINIMUM})")
        axis.set_xlabel("Curso y presentación")
        axis.set_ylabel("Tasa de riesgo académico (%)")
        axis.tick_params(axis="x", rotation=45)
        self._save_figure(figure, "academic_risk_by_course_presentation.png")
        return final_result, risk, engagement, course, score_missingness

    def write_report(
        self,
        snapshot: pd.DataFrame,
        quality: pd.DataFrame,
        missingness: pd.DataFrame,
        collinearity: pd.DataFrame,
        final_result: pd.DataFrame,
        risk: pd.DataFrame,
        engagement: pd.DataFrame,
        course: pd.DataFrame,
        score_missingness: pd.DataFrame,
        bivariate_scores: pd.DataFrame,
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
        risk_total = int(risk.loc[risk["academic_risk"] == RISK_LABEL, "n_total"].iloc[0])
        risk_share = float(risk.loc[risk["academic_risk"] == RISK_LABEL, "share"].iloc[0])
        outcome_summary = ", ".join(
            f"{row.final_result}: {int(row.n_total):,}/{len(self.df):,} ({row.share:.2%})"
            for row in final_result.itertuples()
        )
        missingness_summary = "; ".join(
            f"{row.final_result}: {int(row.n_score_missing):,}/{int(row.n_total):,} ({row.score_missing_rate:.2%})"
            for row in score_missingness.itertuples()
        )
        bivariate_observed = int(bivariate_scores["observed_score_count"].sum() / len(DESCRIPTIVE_ONLY_COLUMNS))
        bivariate_missing = int(bivariate_scores["missing_score_count"].sum() / len(DESCRIPTIVE_ONLY_COLUMNS))
        cutoff = f"día {self.cutoff_day}" if self.cutoff_day is not None else "corte temporal documentado"

        lines = [
            "# EDA de OULAD: alerta temprana de resultado académico adverso",
            "",
            f"Este análisis descriptivo usa el artefacto de matrículas con variables observadas hasta el {cutoff}. La hipótesis principal es identificar señales tempranas de resultado académico adverso: `Withdrawn` o `Fail`, frente a `Pass` o `Distinction`.",
            "",
            "## Hallazgos de resultado y riesgo",
            "",
            f"Distribución de `final_result`: {outcome_summary}.",
            "",
            "![Distribución del resultado final](figures/final_result_distribution.png)",
            "",
            "La distribución de `total_clicks` hasta el corte se presenta por resultado final en `total_clicks_by_final_result_boxplot_summary.csv`. La caja representa Q1 a Q3 y la mediana; los bigotes muestran los valores no atípicos según 1.5 IQR. Es una comparación descriptiva y no demuestra causalidad.",
            "",
            "![Distribución de clics por resultado final](figures/total_clicks_by_final_result_boxplot.png)",
            "",
            f"El grupo descriptivo `{RISK_LABEL}` reúne `Withdrawn` y `Fail`: {risk_total:,}/{len(self.df):,} matrículas ({risk_share:.2%}). `{NON_RISK_LABEL}` reúne `Pass` y `Distinction`. Esta agrupación se usa solo para describir resultados y no es una variable candidata del modelo.",
            "",
            f"La tasa descriptiva de riesgo académico es {first_quantile['academic_risk_rate']:.2%} en Q1 y {last_quantile['academic_risk_rate']:.2%} en Q4 de interacción. Cada tasa conserva su denominador: Q1 {int(first_quantile['n_academic_risk']):,}/{int(first_quantile['n_total']):,}; Q4 {int(last_quantile['n_academic_risk']):,}/{int(last_quantile['n_total']):,}.",
            "",
            "![Riesgo académico por interacción](figures/academic_risk_by_engagement_quartile.png)",
            "",
            f"Por curso y presentación se suprimen las tasas de {suppressed_groups} grupos con n inferior a {SPARSE_GROUP_MINIMUM}; las tasas visibles se acompañan de su denominador en `academic_risk_by_course_presentation.csv`.",
            "",
            "![Riesgo académico por curso y presentación](figures/academic_risk_by_course_presentation.png)",
            "",
            "## Disponibilidad de puntuaciones",
            "",
            f"Faltantes de `{SCORE_COLUMN}` por resultado final: {missingness_summary}. Los porcentajes se calculan dentro de cada `final_result`; describen disponibilidad de datos y no explican ni causan el resultado.",
            "",
            "![Faltantes de puntuación por resultado final](figures/score_missingness_by_final_result.png)",
            "",
            "![Valores faltantes y puntuaciones observadas](figures/missingness_and_observed_scores.png)",
            "",
            "## Puntuaciones observadas por características sensibles",
            "",
            f"`bivariate_score_boxplots_summary.csv` conserva {bivariate_observed:,} puntuaciones observadas y {bivariate_missing:,} faltantes de `{SCORE_COLUMN}` para cada característica. El panel muestra solo las distribuciones de puntuaciones observadas; los conteos `n` en las etiquetas y la columna `missing_score_count` mantienen explícita la disponibilidad desigual del resultado.",
            "",
            "![Puntuaciones observadas por características sensibles](figures/bivariate_score_boxplots.png)",
            "",
            "Este panel es descriptivo, no causal, no permite concluir equidad o inequidad y no es un resultado de selección de variables. `gender`, `age_band`, `highest_education` y `weighted_assessment_score` permanecen excluidos de las variables candidatas del modelo.",
            "",
            "## Cohorte y calidad de datos",
            "",
            f"El artefacto contiene {int(values['enrollment_rows']):,} filas de matrícula y {int(values['columns'])} columnas. Representa a {int(values['unique_students']):,} estudiantes en {int(values['courses'])} cursos y {int(values['presentations'])} presentaciones; la clave de matrícula es `{', '.join(KEY_COLUMNS)}`. {zero_vle:,} filas tienen cero clics en el VLE antes del corte.",
            "",
            f"`{SCORE_COLUMN}` se observa en {int(score['available_count']):,}/{len(self.df):,} filas ({score['available_count'] / len(self.df):.2%}) y falta en {int(score['missing_count']):,}/{len(self.df):,} ({score['missing_share']:.2%}). `imd_band` falta en {int(imd['missing_count']):,}/{len(self.df):,} filas ({imd['missing_share']:.2%}) y se muestra con la etiqueta `{MISSING_IMD_LABEL}`.",
            "",
            "![Interacción de la cohorte](figures/cohort_engagement_total_clicks.png)",
            "",
            "![Perfil categórico de la cohorte](figures/categorical_cohort_profile.png)",
            "",
            "## Distribuciones univariadas de variables candidatas",
            "",
            f"Los paneles incluyen {len(self.numeric_candidate_features)} variables numéricas y {len(self.categorical_candidate_features)} categóricas candidatas. Excluyen las claves de matrícula, identificadores numéricos, `{', '.join(TARGET_COLUMNS)}`, la variable derivada `academic_risk` y las características sensibles `{', '.join(DESCRIPTIVE_ONLY_COLUMNS)}`; `final_result` se describe únicamente en el EDA de resultados.",
            "",
            "![Histogramas de variables numéricas candidatas](figures/univariate_numeric_histograms.png)",
            "",
            "El resumen numérico por variable está disponible en `univariate_numeric_summary.csv`; el resumen de conteos y proporciones categóricas, incluidos los faltantes como `Sin dato`, está en `univariate_categorical_summary.csv`.",
            "",
            "![Frecuencias de variables categóricas candidatas](figures/univariate_categorical_frequencies.png)",
            "",
            "Estas distribuciones son descriptivas: no prueban causalidad ni constituyen por sí solas una selección de variables.",
            "",
            "## Preparación para el modelado y exclusiones por fuga de información",
            "",
            f"El EDA considera las {int(values['candidate_features'])} columnas que no son claves, resultados o características sensibles como variables candidatas. `{', '.join(TARGET_COLUMNS)}`, la agrupación descriptiva `academic_risk` y `{', '.join(DESCRIPTIVE_ONLY_COLUMNS)}` no se usan como variables de modelo ni aparecen en correlaciones o comprobaciones de colinealidad.",
            "",
            f"Las correlaciones de Spearman usan únicamente variables candidatas numéricas. El resumen de colinealidad identifica {near_duplicates} {near_duplicate_label} de variables candidatas con correlación absoluta de Spearman de al menos 0.95; es un cribado, no una selección de variables.",
            "",
            "Estas comparaciones son descriptivas: no demuestran causalidad, no evalúan un modelo predictivo y no sustituyen una evaluación temporal con datos reservados.",
            "",
            "![Correlaciones entre variables candidatas](figures/candidate_feature_spearman_correlations.png)",
        ]
        report = self.output_dir / "eda_report.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.generated_files.append(report)
        return report

    def run_all(self) -> list[Path]:
        """Run all leakage-safe descriptive analyses and return their manifest."""
        self.validate_contract()
        self._remove_obsolete_generated_artifacts()
        snapshot = self.cohort_snapshot()
        quality = self.data_quality()
        self.categorical_profile()
        self.numeric_univariate_profile()
        self.categorical_univariate_profile()
        missingness = self.missingness_and_scores()
        correlation = self.candidate_spearman_correlations()
        collinearity = self.collinearity_summary(correlation)
        self.total_clicks_by_final_result()
        bivariate_scores = self.bivariate_score_boxplots()
        final_result, risk, engagement, course, score_missingness = self.academic_risk_analysis()
        self.write_report(snapshot, quality, missingness, collinearity, final_result, risk, engagement, course, score_missingness, bivariate_scores)
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
