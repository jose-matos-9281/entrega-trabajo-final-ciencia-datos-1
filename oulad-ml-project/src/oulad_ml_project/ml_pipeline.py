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

import pandas as pd
import warnings
import json
from pathlib import Path
from .preprocessing import DataPreprocessor
from .train_ml import ModelTrainer
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Configuración de rutas del proyecto
# ------------------------------------------------------------


# ------------------------------------------------------------
# CLASE 4: ORQUESTADOR DEL PIPELINE (OSEMN)
# ------------------------------------------------------------
class MLPipeline:
    """
    Orquestador principal del pipeline OSEMN.

    Flujo completo:
      1. OBTAIN: Cargar datos desde CSV
      2. SCRUB: EDA + preprocesamiento (missing, encoding)
      3. EXPLORE: Visualizaciones y estadísticas
      4. MODEL: Entrenamiento de todos los modelos
      5. INTERPRET: Métricas, gráficos, CSV outputs
    """

    def __init__(self, data_path: Path, data_dir: Path, output_dir: Path):
        self.data_path = data_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.df = None
        self.preprocessor = None
        self.trainer = None

    def load_data(self):
        """Carga el dataset desde el archivo CSV."""
        print("=" * 60)
        print("PROYECTO OULAD - MACHINE LEARNING")
        print("=" * 60)
        self.df = pd.read_csv(self.data_path)
        print(f"\n[OBTAIN] Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return self.df


    def preprocess(self):
        """Preprocesa los datos: missing values + encoding."""
        print("\n[SCRUB] Preprocesando datos...")
        self.preprocessor = DataPreprocessor(self.df)
        self.preprocessor.handle_missing()
        self.preprocessor.encode_ordinal()
        return self.preprocessor.df

    def prepare_model_data(self):
        """
        Prepara los datos para el modelado (features + targets).

        Uses only the real OULAD artifact contract.
        """
        target_cols = ['passed', 'performance_tier', 'weighted_assessment_score']
        X, y_sets, feature_cols = self.preprocessor.prepare_features(target_cols)
        print(f"[MODEL] Características seleccionadas: {len(feature_cols)}")
        print(f"  Features: {feature_cols}")
        return X, y_sets, feature_cols

    def train_models(self):
        """Train, select, and persist the leakage-safe binary risk champion."""
        metadata_path = self.data_dir / "oulad_training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cutoff_day = metadata.get("cutoff_day")
        print("\n[MODEL] Training risk-classifier candidates...")
        self.trainer = ModelTrainer.for_risk_training(self.df, self.output_dir / "artifacts", cutoff_day)
        artifacts = self.trainer.train_risk_champion()
        self.results = artifacts.as_dict()
        return self.results

    def run(self):
        """
        Ejecuta el pipeline OSEMN completo.

        Returns:
            Dict con resultados de todos los modelos
        """
        self.load_data()
        print(f"  Distribución target (passed):\n{self.df['passed'].value_counts(normalize=True)}")
        print(f"  Distribución target (performance_tier):\n{self.df['performance_tier'].value_counts(normalize=True)}")
        self.train_models()
        print("\n" + "=" * 60)
        print("PROYECTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        return self.results


# ------------------------------------------------------------
# PUNTO DE ENTRADA
# ------------------------------------------------------------
def main():
    """
    Ejecuta el pipeline completo:
      1. Carga datos
      2. EDA completo con gráficos
      3. Preprocesamiento
      4. 12 modelos + clustering
      5. Exportación de CSVs y gráficos
    """
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" 
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir/".." / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "oulad_training_full.csv"
    pipeline = MLPipeline(data_path, data_dir, output_dir)
    pipeline.run()
