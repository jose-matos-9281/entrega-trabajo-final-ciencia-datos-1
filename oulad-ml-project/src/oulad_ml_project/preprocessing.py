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

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Configuración de rutas del proyecto
# ------------------------------------------------------------

# ------------------------------------------------------------
# CLASE 2: PREPROCESAMIENTO DE DATOS
# ------------------------------------------------------------
class DataPreprocessor:
    """
    Preprocesamiento de datos: manejo de missing values y codificación.

    Según la rúbrica (1 pt):
      - Identificación y tratamiento de valores faltantes (justificado)
      - Codificación de variables ordinales con OrdinalEncoder
      - Explicit ordinal mappings; nominal values remain raw for training pipelines
      - Prevención de fuga de datos (data leakage)

    Estrategia de imputación:
       - Numéricas restantes: mediana
      - Categóricas: moda (valor más frecuente)
    """

    def __init__(self, df):
        self.df = df.copy()
        self.ordinal_mappings = {}

    def handle_missing(self):
        """
        Maneja los valores faltantes con estrategias justificadas.

        Justificación:
           - Variables numéricas generales: mediana (robusta a outliers)
          - Variables categóricas: moda (valor más representativo)
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns

        target_columns = {'final_result', 'passed', 'performance_tier', 'weighted_assessment_score'}

        # Only features are imputed. Targets retain missingness from the source.
        for col in num_cols:
            if col not in target_columns and self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # Categóricas restantes: imputación con moda
        for col in cat_cols:
            if col not in target_columns and self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    def encode_ordinal(self):
        """
        Encode only variables with an explicit, valid ordinal order.

        Nominal features remain raw. Production pipelines use one-hot encoding
        fitted on the training partition rather than arbitrary label integers.
        """
        ordinal_cols = {
            'highest_education': ['No Formal', 'Lower Secondary', 'Upper Secondary', 'Bachelor', 'Master'],
            'education_level': ['No Formal', 'Lower Secondary', 'Upper Secondary', 'Bachelor', 'Master'],
            'age_band': ['0-35', '35-55', '55-']
        }
        for col, categories in ordinal_cols.items():
            if col in self.df.columns:
                non_null = self.df[col].notna()
                valid_mask = self.df[col].isin(categories)
                self.df.loc[non_null & valid_mask, col + '_encoded'] = pd.Categorical(
                    self.df.loc[non_null & valid_mask, col], categories=categories, ordered=True
                ).codes
                self.df[col + '_encoded'] = self.df[col + '_encoded'].fillna(0).astype(int)
                self.ordinal_mappings[col] = {c: i for i, c in enumerate(categories)}

        return self.df

    def prepare_features(self, target_cols, exclude_cols=None):
        """
        Prepara las matrices de características (X) y targets (y).

        Previene fuga de datos (data leakage) excluyendo:
           - target columns and enrollment identifiers
          - id_student (sin valor predictivo)

        Args:
            target_cols: Lista de columnas target a extraer
            exclude_cols: Columnas adicionales a excluir

        Returns:
            X: DataFrame de características
            y_sets: Dict con cada target
            feature_cols: Lista de nombres de características
        """
        if exclude_cols is None:
            exclude_cols = ['id_student', 'code_presentation']
        exclude_cols = exclude_cols + target_cols
        exclude_cols += ['code_module', 'code_presentation']

        feature_cols = [c for c in self.df.columns
                       if c not in exclude_cols
                       and self.df[c].dtype in [np.int64, np.float64, int, float]
                       and c not in self.df.select_dtypes(include=['object', 'category']).columns
                       and not (c.endswith('_enc') and not c.endswith('_encoded')
                                and c.replace('_enc', '_encoded') in self.df.columns)]

        X = self.df[feature_cols].copy()
        y_sets = {}
        for t in target_cols:
            if t in self.df.columns:
                y_sets[t] = self.df[t]
        return X, y_sets, feature_cols
