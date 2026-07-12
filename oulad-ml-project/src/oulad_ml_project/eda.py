"""
=============================================================================
  PIPELINE COMPLETO DE MACHINE LEARNING - OULAD
=============================================================================
 Proyecto Final Colaborativo - Machine Learning sobre OULAD
 Curso: Data Analysis | Fecha: Julio 2026

 Este módulo implementa el pipeline OSEMN completo bajo POO:
   O - Obtain:  Carga del dataset
   S - Scrub:   Limpieza, EDA, manejo de missing values
   E - Explore: Análisis univariado, bivariado, correlacional
   M - Model:   Entrenamiento de modelos supervisados y no supervisados
   N - iNterpret: Métricas, feature importance, gráficos, CSVs

 Estructura de clases:
   - ExploratoryDataAnalysis: EDA completo con gráficos y estadísticas
   - DataPreprocessor:        Limpieza, imputación, codificación ordinal
   - ModelTrainer:            Entrenamiento de 12 modelos + clustering
   - MLPipeline:              Orquestador que conecta todo el flujo


=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuración de rutas del proyecto
# ------------------------------------------------------------


# ------------------------------------------------------------
# CLASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ------------------------------------------------------------
class ExploratoryDataAnalysis:
    """
    Realiza el Análisis Exploratorio de Datos completo.

    Cubre según la rúbrica (4 pts):
      - Estadísticas descriptivas (media, std, min, max, kurtosis, skewness)
      - Análisis univariado (histogramas numéricos, barras categóricas)
      - Análisis bivariado (scatter plots, box plots por categoría)
      - Matriz de correlación (heatmap)
      - Gráficos de dispersión
      - Kurtosis y asimetría
      - Box plots
      - Missing value analysis

    Atributos:
        df: DataFrame original
        numeric_cols: Columnas numéricas (excluye id_student)
        categorical_cols: Columnas categóricas/objeto
        fig_dir: Directorio donde se guardan las figuras
    """

    def __init__(self, df, data_dir: Path, output_dir: Path):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        # Excluimos columnas de ID del análisis numérico
        self.numeric_cols = [c for c in self.numeric_cols if "id_" not in c]
        self.fig_dir = output_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.data_dir = data_dir

    def descriptive_stats(self):
        """
        Calcula estadísticas descriptivas de todas las columnas numéricas.

        Incluye:
          - count, mean, std, min, 25%, 50%, 75%, max
          - kurtosis: medida de la "cola" de la distribución
          - skewness: medida de asimetría de la distribución
          - missing: conteo de valores faltantes

        Guarda: output/descriptive_statistics.csv
        """
        print("\n========== ESTADÍSTICAS DESCRIPTIVAS ==========")
        stats = self.df[self.numeric_cols].describe().T
        stats["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        stats["asimetria"] = self.df[self.numeric_cols].skew()
        stats["missing"] = self.df[self.numeric_cols].isna().sum()
        stats.to_csv(self.output_dir / "descriptive_statistics.csv")
        print(stats.to_string())
        return stats

    def univariate_analysis(self):
        """
        Análisis univariado con histogramas (numéricas) y barras (categóricas).

        Genera:
          - output/figures/univariate_histograms.png (hasta 12 numéricas)
          - output/figures/univariate_categorical.png (hasta 6 categóricas)
        """
        print("\n========== ANÁLISIS UNIVARIADO ==========")
        # Histogramas para variables numéricas
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        for i, col in enumerate(self.numeric_cols[:12]):
            self.df[col].hist(bins=30, ax=axes[i], edgecolor="black")
            axes[i].set_title(col)
        plt.tight_layout()
        plt.savefig(self.fig_dir / "univariate_histograms.png")
        plt.close()

        # Barras para variables categóricas
        cat_cols = self.categorical_cols[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            self.df[col].value_counts().plot(kind="bar", ax=axes[i], edgecolor="black")
            axes[i].set_title(col)
            axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(self.fig_dir / "univariate_categorical.png")    
        plt.close()
        print("Gráficos univariados guardados.")

    def bivariate_analysis(self):
        """
        Análisis bivariado con scatter plots y box plots.

        Scatter plots: 6 gráficos de dispersión contra weighted_assessment_score
        Box plots: weighted_assessment_score segmentado por variables demográficas

        Genera:
          - output/figures/bivariate_scatter.png
          - output/figures/bivariate_boxplots.png
        """
        print("\n========== ANÁLISIS BIVARIADO ==========")
        target = "weighted_assessment_score"
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        biv_cols = [c for c in self.numeric_cols if c != target][:6]
        for i, col in enumerate(biv_cols):
            axes[i].scatter(self.df[col], self.df[target], alpha=0.4)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target)
        plt.tight_layout()
        plt.savefig(self.fig_dir / "bivariate_scatter.png")
        plt.close()

        # Box plots para variables categóricas clave
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(["gender", "age_band", "highest_education"]):
            self.df.boxplot(column=target, by=col, ax=axes[i])
            axes[i].set_title(f"{target} por {col}")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "bivariate_boxplots.png")
        plt.close()
        print("Gráficos bivariados guardados.")

    def correlation_analysis(self):
        """
        Matriz de correlación entre todas las variables numéricas.

        Genera:
          - output/figures/correlation_matrix.png (heatmap)
          - output/correlation_matrix.csv

        Interpretación:
          - Valores cercanos a 1: correlación positiva fuerte
          - Valores cercanos a -1: correlación negativa fuerte
          - Valores cercanos a 0: sin correlación lineal
        """
        print("\n========== ANÁLISIS DE CORRELACIÓN ==========")
        corr = self.df[self.numeric_cols].corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Matriz de Correlación")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "correlation_matrix.png")    
        plt.close()
        corr.to_csv(self.output_dir / "correlation_matrix.csv")
        print("Matriz de correlación guardada.")

    def missing_value_analysis(self):
        """
        Analiza y visualiza los valores faltantes en el dataset.

        Genera:
          - output/missing_values.csv (conteo y porcentaje)
          - output/figures/missing_values.png (gráfico de barras)

        Returns:
            DataFrame con columnas: missing_count, missing_pct
        """
        print("\n========== ANÁLISIS DE VALORES FALTANTES ==========")
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame(
            {"conteo_faltantes": missing, "porcentaje": missing_pct}
        )
        missing_df = missing_df[missing_df["conteo_faltantes"] > 0].sort_values(
            "conteo_faltantes", ascending=False
        )
        print(missing_df.to_string())
        missing_df.to_csv(self.output_dir / "missing_values.csv")

        plt.figure(figsize=(10, 6))
        missing_df["porcentaje"].plot(kind="barh")
        plt.xlabel("Porcentaje de Valores Faltantes (%)")
        plt.title("Valores Faltantes por Columna")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "missing_values.png")
        plt.close()
        return missing_df

    def run_all(self):
        """Ejecuta todos los análisis de EDA en secuencia."""
        self.descriptive_stats()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.correlation_analysis()
        self.missing_value_analysis()
        print("\nEDA completado. Todas las figuras guardadas en", self.fig_dir)
