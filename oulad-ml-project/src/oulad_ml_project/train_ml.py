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

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
                             ConfusionMatrixDisplay, silhouette_score)
from sklearn.cluster import KMeans
import warnings
from pathlib import Path
from .risk_model import TrainingArtifacts, train_risk_champion
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Configuración de rutas del proyecto
# ------------------------------------------------------------


# ------------------------------------------------------------
# CLASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ------------------------------------------------------------
class ModelTrainer:
    """
    Entrenamiento de modelos supervisados y no supervisados.

    Según la rúbrica (8 pts):
      - Mínimo 3 algoritmos supervisados por tarea
      - Variables dicotómica, ordinal e intervalo/razón
      - Tendencias no-supervisadas (KMeans)
      - Cálculo manual de F1-score (TP, FP, TN, FN)
      - Feature importance
      - CSVs de predicciones (y_test, y_pred)
      - Matrices de confusión
      - Métricas: precision_macro, recall_macro, f1_macro,
                  accuracy, roc_auc, mse, r2

    Modelos por tipo:
      - Binario (passed): LogisticRegression, RandomForest, GradientBoosting
      - Ordinal (performance_tier): DecisionTree, RandomForest, GradientBoosting
       - Regresión (weighted_assessment_score): LinearRegression, RandomForestRegressor, KNN
      - No supervisado: KMeans (k=3)
    """

    def __init__(self, X=None, y_dict=None, output_dir: Path | None = None, student_groups=None,
                 *, training_frame: pd.DataFrame | None = None, cutoff_day: int | None = None):
        """Create an auxiliary-analysis trainer or the production risk-training facade.

        The legacy multi-target methods remain available for exploratory analysis,
        but only ``train_risk_champion`` publishes the inference champion bundle.
        """
        self.X = X
        self.y_dict = y_dict
        self.output_dir = output_dir
        self.student_groups = student_groups.loc[X.index] if student_groups is not None and X is not None else None
        self.training_frame = training_frame
        self.cutoff_day = cutoff_day
        self.results = {}
        self.models = {}

    @classmethod
    def for_risk_training(cls, frame: pd.DataFrame, artifacts_dir: Path, cutoff_day: int) -> "ModelTrainer":
        """Construct the sole production facade for the leakage-safe risk champion."""
        return cls(output_dir=artifacts_dir, training_frame=frame, cutoff_day=cutoff_day)

    def train_risk_champion(self) -> TrainingArtifacts:
        """Publish the leakage-safe champion and its reproducible training evidence."""
        if self.training_frame is None or self.output_dir is None or self.cutoff_day is None:
            raise ValueError("risk training requires frame, artifacts_dir, and cutoff_day")
        artifacts = train_risk_champion(self.training_frame, self.output_dir, self.cutoff_day)
        return self._add_risk_visualizations(artifacts)

    def _add_risk_visualizations(self, artifacts: TrainingArtifacts) -> TrainingArtifacts:
        """Add non-interactive visual evidence without changing champion selection."""
        report = json.loads(artifacts.report.read_text(encoding="utf-8"))
        manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
        candidates = pd.read_csv(artifacts.metrics)
        holdout = pd.read_csv(artifacts.evaluation_predictions)
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        version = artifacts.model.name.removesuffix("-passed_model.joblib")

        metrics_plot = figures_dir / f"{version}-candidate_metrics.png"
        plot_metrics = ["risk_recall", "risk_precision", "risk_f1", "accuracy", "roc_auc"]
        candidates.set_index("model")[plot_metrics].plot(kind="bar", figsize=(11, 6))
        plt.ylabel("Cross-validation mean")
        plt.title("Risk-model candidate comparison")
        plt.tight_layout()
        plt.savefig(metrics_plot, dpi=150)
        plt.close()

        confusion_plot = figures_dir / f"{version}-champion_confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(
            holdout["passed_actual"], holdout["prediction_passed"], labels=[0, 1], cmap="Blues"
        )
        plt.title(f"Champion holdout confusion matrix: {report['champion']['name']}")
        plt.tight_layout()
        plt.savefig(confusion_plot, dpi=150)
        plt.close()

        model = joblib.load(artifacts.model)
        classifier = model.named_steps["classifier"]
        feature_importance: dict[str, object] = {
            "status": "unavailable",
            "reason": f"{classifier.__class__.__name__} does not expose coefficients or feature_importances_",
        }
        importance_plot = None
        if hasattr(classifier, "coef_") or hasattr(classifier, "feature_importances_"):
            names = model.named_steps["preprocessor"].get_feature_names_out()
            values = np.abs(classifier.coef_).ravel() if hasattr(classifier, "coef_") else classifier.feature_importances_
            top = pd.Series(values, index=names).sort_values().tail(15)
            importance_plot = figures_dir / f"{version}-champion_feature_importance.png"
            top.plot(kind="barh", figsize=(10, 7))
            plt.xlabel("Absolute coefficient" if hasattr(classifier, "coef_") else "Feature importance")
            plt.title(f"Champion feature importance: {report['champion']['name']}")
            plt.tight_layout()
            plt.savefig(importance_plot, dpi=150)
            plt.close()
            feature_importance = {"status": "generated", "path": str(importance_plot.relative_to(self.output_dir))}

        artifact_paths = {
            "model": artifacts.model.name,
            "manifest": artifacts.manifest.name,
            "report": artifacts.report.name,
            "candidate_metrics_csv": artifacts.metrics.name,
            "holdout_predictions_csv": artifacts.evaluation_predictions.name,
            "candidate_metrics_plot": str(metrics_plot.relative_to(self.output_dir)),
            "champion_confusion_matrix_plot": str(confusion_plot.relative_to(self.output_dir)),
            "champion_feature_importance": feature_importance,
        }
        report["artifacts"] = artifact_paths
        report["limitations"] = [
            "Candidate metrics are GroupKFold means on the training partition, not holdout metrics.",
            "The holdout is evaluated once after selection and must not be used for model selection.",
            "Feature importance is only reported when the selected classifier exposes coefficients or feature_importances_.",
        ]
        artifacts.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        manifest["training_artifacts"] = artifact_paths
        artifacts.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return artifacts

    def split_data(self, target_name, target_col=None):
        """
        Divide datos en entrenamiento (75%) y prueba (25%).

        Mantiene cada estudiante exclusivamente en entrenamiento o prueba.
        No se estratifica porque GroupShuffleSplit no ofrece estratificación
        agrupada y una estratificación por fila reintroduciría leakage.

        Returns:
            X_train, X_test, y_train, y_test, label_encoder (o None)
        """
        y = self.y_dict[target_name]
        if target_col and target_col in self.y_dict:
            y = self.y_dict[target_col]
        y_enc, le = self.encode_ordinal_target(y)
        X_train, X_test, y_train, y_test = self.split_grouped(self.X, y_enc)
        return X_train, X_test, y_train, y_test, le

    def split_grouped(self, X, y):
        """Split rows by student identity while keeping identities out of X."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        groups = self.student_groups.loc[X.index]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_indices, test_indices = next(splitter.split(X, y, groups))
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

    def encode_ordinal_target(self, y):
        """
        Codifica targets categóricos/ordinales a valores numéricos.
        """
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            return y_enc, le
        return y, None

    def train_binary(self):
        """
        Modelo dicotómico: predicción de 'passed' (aprobado/reprobado).

        Algoritmos:
          - LogisticRegression: modelo lineal interpretable
          - RandomForest: ensemble de árboles, captura no linealidades
          - GradientBoosting: boosting secuencial, alta precisión

        Métricas:
          - accuracy, precision_macro, recall_macro, f1_macro, roc_auc
          - TP, FP, TN, FN (cálculo manual de F1-score)
          - Feature importance (coef_ o feature_importances_)
        """
        print("\n========== CLASIFICACIÓN BINARIA: Aprobado ==========")
        X_train, X_test, y_train, y_test, _ = self.split_data('passed', 'passed')

        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        binary_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            mse_val = mean_squared_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            results = {
                'model': name,
                'type': 'binary',
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                'mse': mse_val,
                'r2': r2_val,
                'msePI2': mse_val,
                'r2PI2': r2_val,
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                'f1_score': f1_score(y_test, y_pred)
            }
            binary_results[name] = results
            print(f"{name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}, AUC={results['roc_auc']:.4f}")
            print(f"  Matriz de Confusión: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

            if hasattr(model, 'feature_importances_'):
                results['feature_importances'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                results['feature_importances'] = model.coef_[0].tolist()
            self.models[f'binary_{name}'] = model

        self.results['binary'] = binary_results
        return binary_results

    def train_ordinal(self):
        """
        Modelo ordinal: predicción de 'performance_tier' (Fail/Low/Medium/High).

        Algoritmos:
          - DecisionTree: interpretable, propenso a overfitting
          - RandomForest: ensemble robusto
          - GradientBoosting: boosting con manejo ordinal

        Nota: Se usa MSE como métrica adicional porque los niveles tienen orden.
        """
        print("\n========== CLASIFICACIÓN ORDINAL: Nivel de Rendimiento ==========")
        y_original = self.y_dict.get('performance_tier', None)
        if y_original is None:
            print("performance_tier no encontrado, saltando...")
            return {}
        y_enc = y_original if y_original.dtype in [np.int64, np.float64, int, float] \
                          else LabelEncoder().fit_transform(y_original.astype(str))

        X_train, X_test, y_train, y_test = self.split_grouped(self.X, y_enc)

        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        ordinal_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results = {
                'model': name,
                'type': 'ordinal',
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            ordinal_results[name] = results
            print(f"{name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}, MSE={results['mse']:.4f}")

            if hasattr(model, 'feature_importances_'):
                results['feature_importances'] = model.feature_importances_.tolist()
            self.models[f'ordinal_{name}'] = model

        self.results['ordinal'] = ordinal_results
        return ordinal_results

    def train_regression(self):
        """
        Modelo de regresión: predicción de 'weighted_assessment_score'.

        Algoritmos:
          - LinearRegression: modelo lineal base
          - RandomForestRegressor: ensemble no lineal
          - KNNRegressor: basado en vecinos cercanos

        Métricas: MSE (Error Cuadrático Medio), R² (Coeficiente de Determinación)
        """
        print("\n========== REGRESIÓN: Nota Final ==========")
        y = self.y_dict.get('weighted_assessment_score', None)
        if y is None:
            print("weighted_assessment_score no encontrado, saltando...")
            return {}

        scored_rows = y.notna()
        X = self.X.loc[scored_rows]
        y = y.loc[scored_rows]

        X_train, X_test, y_train, y_test = self.split_grouped(X, y)

        models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }

        regression_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results = {
                'model': name,
                'type': 'regression',
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }
            regression_results[name] = results
            print(f"{name}: MSE={results['mse']:.6f}, R²={results['r2']:.4f}")

            if hasattr(model, 'feature_importances_'):
                results['feature_importances'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                coefs = model.coef_
                results['feature_importances'] = coefs.tolist() if len(coefs.shape) == 1 else coefs[0].tolist()
            self.models[f'regression_{name}'] = model

        self.results['regression'] = regression_results
        return regression_results

    def run_unsupervised(self):
        """
        Análisis no supervisado con KMeans clustering.

        Segmenta estudiantes en 3 clusters. Evalúa con:
          - Inercia (suma de distancias intra-cluster)
          - Silhouette Score (cohesión vs separación)
        """
        print("\n========== NO SUPERVISADO: KMeans Clustering ==========")
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(self.X)
        sil = silhouette_score(self.X, clusters)
        self.results['unsupervised'] = {
            'clusters': clusters.tolist(),
            'inertia': kmeans.inertia_,
            'silhouette_score': sil
        }
        print(f"Inercia KMeans: {kmeans.inertia_:.2f}")
        print(f"Silhouette Score: {sil:.4f}")
        return clusters

    def save_predictions_csv(self):
        """
        Exporta predicciones a CSV: cada modelo produce un archivo
        con las columnas y_test (real) y y_pred (predicho).

        Archivos generados (ejemplos):
          - binary_LogisticRegression_predictions.csv
          - ordinal_RandomForest_predictions.csv
          - regression_LinearRegression_predictions.csv
        """
        print("\n========== GUARDANDO PREDICCIONES CSV ==========")
        for result_type, models_dict in self.results.items():
            if isinstance(models_dict, dict):
                for model_name, results in models_dict.items():
                    if isinstance(results, dict) and 'y_pred' in results and 'type' in results:
                        fname = f"{result_type}_{model_name}_predictions.csv"
                        y_test_list = results.get('y_test', [])
                        y_pred_list = results['y_pred']
                        min_len = min(len(y_test_list), len(y_pred_list))
                        if min_len > 0:
                            df_out = pd.DataFrame({
                                'y_test': y_test_list[:min_len],
                                'y_pred': y_pred_list[:min_len]
                            })
                            df_out.to_csv(os.path.join(self.output_dir, fname), index=False)
                            print(f"  Guardado {fname} ({len(df_out)} filas)")

    def save_metrics_summary(self):
        """
        Exporta un resumen de todas las métricas a CSV.

        Métricas incluidas:
          - accuracy, precision_macro, recall_macro, f1_macro
          - roc_auc (solo clasificación binaria)
          - mse, r2 (solo regresión y ordinal)
          - f1_score (binaria)
          - TP, FP, TN, FN (binaria)
        """
        rows = []
        for model_type, models_dict in self.results.items():
            if isinstance(models_dict, dict):
                for model_name, results in models_dict.items():
                    if not isinstance(results, dict):
                        continue
                    row = {'model_name': model_name, 'type': results.get('type', model_type)}
                    for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                                'roc_auc', 'mse', 'r2', 'msePI2', 'r2PI2', 'f1_score']:
                        row[key] = results.get(key, None)
                    row['TP'] = results.get('TP', None)
                    row['FP'] = results.get('FP', None)
                    row['TN'] = results.get('TN', None)
                    row['FN'] = results.get('FN', None)
                    rows.append(row)

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(os.path.join(self.output_dir, 'metrics_summary.csv'), index=False)
        print(f"\nResumen de métricas guardado: {len(rows)} modelos")
        print(df_metrics.to_string())
        return df_metrics

    def plot_feature_importance(self, feature_names, top_n=10):
        """
        Genera gráficos de importancia de características para cada modelo
        basado en árboles o coeficientes.

        Interpretación:
          - Modelos lineales: valores absolutos de coeficientes
          - Modelos de árboles: reducción de impureza promedio
          - Más alto = más influyente en la predicción
        """
        print("\n========== IMPORTANCIA DE CARACTERÍSTICAS ==========")
        for key, model in self.models.items():
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                fi = model.feature_importances_
                if len(fi) == len(feature_names):
                    idx = np.argsort(fi)[-top_n:]
                    plt.figure(figsize=(10, 6))
                    plt.barh(range(top_n), fi[idx])
                    plt.yticks(range(top_n), [feature_names[i] for i in idx])
                    plt.xlabel('Importancia')
                    plt.title(f'Importancia de Variables - {key}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'figures', f'feature_importance_{key}.png'))
                    plt.close()
                    print(f"  {key}: Top variables guardadas")

    def plot_confusion_matrices(self):
        """
        Genera matrices de confusión para modelos de clasificación binaria.

        Las matrices de confusión muestran:
          - TP (True Positives): aciertos positivos
          - FP (False Positives): falsas alarmas
          - TN (True Negatives): aciertos negativos
          - FN (False Negatives): omisiones
        """
        print("\n========== MATRICES DE CONFUSIÓN ==========")
        for result_type, models_dict in self.results.items():
            if isinstance(models_dict, dict):
                for model_name, results in models_dict.items():
                    if isinstance(results, dict) and 'y_test' in results and 'y_pred' in results:
                        if len(results['y_test']) > 0 and len(results['y_pred']) > 0:
                            if len(np.unique(results['y_test'])) <= 2 and len(np.unique(results['y_pred'])) <= 2:
                                fig, ax = plt.subplots(figsize=(6, 5))
                                ConfusionMatrixDisplay.from_predictions(
                                    results['y_test'], results['y_pred'], ax=ax, cmap='Blues'
                                )
                                plt.title(f'Matriz de Confusión - {result_type}_{model_name}')
                                plt.tight_layout()
                                plt.savefig(os.path.join(self.output_dir, 'figures', f'cm_{result_type}_{model_name}.png'))
                                plt.close()
                                print(f"  Matriz de confusión guardada para {result_type}_{model_name}")

    def run_all_training(self, feature_names):
        """
        Ejecuta todos los entrenamientos y exportaciones en secuencia.

        Orden:
          1. Clasificación binaria (passed)
          2. Clasificación ordinal (performance_tier)
           3. Regresión (weighted_assessment_score)
           4. Clustering no supervisado
           5. Exportar predicciones CSV
           6. Exportar resumen de métricas CSV
           7. Gráficos de importancia de variables
           8. Matrices de confusión
        """
        self.train_binary()
        self.train_ordinal()
        self.train_regression()
        self.run_unsupervised()
        self.save_predictions_csv()
        self.save_metrics_summary()
        self.plot_feature_importance(feature_names)
        self.plot_confusion_matrices()
        print("\n========== TODOS LOS MODELOS ENTRENADOS ==========")
        return self.results
