"""
=============================================================================
 PIPELINE COMPLETO DE MACHINE LEARNING - OULAD + Experimento Kongo
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
from pathlib import Path
from .eda import ExploratoryDataAnalysis
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
        print("PROYECTO OULAD + KONGO - MACHINE LEARNING")
        print("=" * 60)
        self.df = pd.read_csv(self.data_path)
        print(f"\n[OBTAIN] Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return self.df

    def run_eda(self):
        """Ejecuta el Análisis Exploratorio de Datos completo."""
        print("\n[SCRUB] Ejecutando EDA...")
        eda = ExploratoryDataAnalysis(self.df, self.data_dir, self.output_dir)
        eda.run_all()

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

        Mantiene kongo_pre_test y kongo_post_test como features para
        el modelo específico del experimento Kongo.
        """
        target_cols = ['passed', 'performance_tier', 'final_grade']
        X, y_sets, feature_cols = self.preprocessor.prepare_features(target_cols)
        for col in ['kongo_pre_test', 'kongo_post_test']:
            if col in self.preprocessor.df.columns:
                X[col] = self.preprocessor.df[col].values
                if col not in feature_cols:
                    feature_cols.append(col)
        y_sets['kongo_pre_test'] = self.preprocessor.df['kongo_pre_test']
        y_sets['kongo_post_test'] = self.preprocessor.df['kongo_post_test']
        print(f"[MODEL] Características seleccionadas: {len(feature_cols)}")
        print(f"  Features: {feature_cols}")
        return X, y_sets, feature_cols

    def train_models(self, X, y_sets, feature_cols):
        """Entrena todos los modelos y guarda resultados."""
        print("\n[MODEL] Entrenando modelos...")
        self.trainer = ModelTrainer(X, y_sets, self.output_dir)
        self.results = self.trainer.run_all_training(feature_cols)
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
        self.run_eda()
        self.preprocess()
        X, y_sets, feature_cols = self.prepare_model_data()
        self.train_models(X, y_sets, feature_cols)
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
    data_path = data_dir / "oulad_kongo_full.csv"
    pipeline = MLPipeline(data_path, data_dir, output_dir)
    pipeline.run()