"""
=============================================================================
 GENERADOR DE DATOS SINTÉTICOS - OULAD + Experimento Kongo
=============================================================================
 Proyecto Final Colaborativo - Machine Learning
 Curso: Data Analysis
 Fecha: Julio 2026

 Descripción:
   Este módulo genera datos sintéticos que simulan la estructura del dataset
   OULAD (Open University Learning Analytics Dataset) combinado con un
   experimento controlado de alfabetización digital en la República Democrática
   del Congo (Kongo).

 Hipótesis del proyecto:
   H1 - Los modelos de ML predicen la aprobación/fracaso con precisión >85%
   H2 - La intervención digital en Kongo mejora significativamente el rendimiento

 Variables generadas:
   - Demográficas: id_student, age_band, gender, region, education_level
   - Curso: code_module, code_presentation, num_of_prev_attempts, studied_credits
   - Interacción: total_clicks, days_from_deadline, score
   - Experimento Kongo: is_kongo, treatment, kongo_pre_test, kongo_post_test,
                        digital_access, internet_speed_mbps
   - Targets: passed (binario), performance_tier (ordinal), final_grade (razón)
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

class OULADDataGenerator:
    """
    Generador de datos sintéticos que replica la estructura del dataset OULAD
    y añade un experimento controlado en la región del Kongo.

    Atributos:
        n (int): Número de estudiantes a generar (default 2000)
        rs (int): Semilla aleatoria para reproducibilidad
    """

    def __init__(self, n_students=2000, random_state=42):
        """
        Inicializa el generador con el número de estudiantes y semilla.

        Args:
            n_students: Cantidad de registros sintéticos a crear
            random_state: Semilla para numpy.random
        """
        self.n = n_students
        self.rs = random_state
        np.random.seed(random_state)

    def generate_student_demographics(self):
        """
        Genera datos demográficos de los estudiantes.

        Variables:
            age_band: Rango etario (0-35, 35-55, 55-)
            gender: Género (M/F)
            region: Región geográfica (incluye Kongo_Urban y Kongo_Rural)
            education_level: Nivel educativo (No Formal, Lower Secondary,
                           Upper Secondary, Bachelor, Master)

        Returns:
            pd.DataFrame: DataFrame con columnas demográficas
        """
        age_band = np.random.choice(['0-35', '35-55', '55-'], size=self.n, p=[0.45, 0.40, 0.15])
        gender = np.random.choice(['M', 'F'], size=self.n, p=[0.48, 0.52])
        region = np.random.choice(
            ['London', 'South East', 'North West', 'East Midlands', 'West Midlands',
             'Yorkshire', 'Scotland', 'Wales', 'Kongo_Urban', 'Kongo_Rural'],
            size=self.n, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.04, 0.15, 0.15]
        )
        education = np.random.choice(
            ['No Formal', 'Lower Secondary', 'Upper Secondary', 'Bachelor', 'Master'],
            size=self.n, p=[0.08, 0.22, 0.35, 0.25, 0.10]
        )
        return pd.DataFrame({
            'id_student': range(self.n),
            'age_band': age_band,
            'gender': gender,
            'region': region,
            'education_level': education
        })

    def generate_course_data(self, demo):
        """
        Genera información académica de los cursos inscritos.

        Variables:
            - code_module: Código del curso (4 opciones simuladas)
            - code_presentation: Presentación del curso (2013J o 2014J)
            - num_of_prev_attempts: Intentos previos (distribución Poisson)
            - studied_credits: Créditos cursados (relacionado con nivel educativo)

        Args:
            demo: DataFrame con datos demográficos (necesita education_level)

        Returns:
            pd.DataFrame: Datos de curso por estudiante
        """
        n = self.n
        courses = ['AAA', 'BBB', 'CCC', 'DDD']
        code_presentations = ['2013J', '2014J']
        course = np.random.choice(courses, size=n)
        code_pres = np.random.choice(code_presentations, size=n)
        num_prev_attempts = np.random.poisson(0.5, size=n).clip(0, 4)

        # Los créditos se simulan en función del nivel educativo
        edu_map = {'No Formal': 0, 'Lower Secondary': 1, 'Upper Secondary': 2, 'Bachelor': 3, 'Master': 4}
        edu_numeric = demo['education_level'].map(edu_map)
        studied_credits = (edu_numeric * 15 + np.random.normal(0, 10, size=n)).clip(30, 180).astype(int)

        return pd.DataFrame({
            'id_student': demo['id_student'],
            'code_module': course,
            'code_presentation': code_pres,
            'num_of_prev_attempts': num_prev_attempts,
            'studied_credits': studied_credits
        })

    def generate_interaction_data(self, demo, course):
        """
        Genera datos de interacción del estudiante con la plataforma LMS.

        Variables:
            - total_clicks: Clics totales en la plataforma
            - days_from_deadline: Días desde registro hasta fecha límite
            - score: Calificación (0-100), afectada por educación y región

        Nota: Los estudiantes del Kongo tienen un efecto negativo simulado
              en la calificación base para reflejar brechas digitales.

        Args:
            demo: DataFrame demográfico (necesita region, education_level)
            course: DataFrame de curso (no usado directamente)

        Returns:
            pd.DataFrame: Datos de interacción
        """
        n = self.n
        clicks = np.random.poisson(200, size=n).clip(0, 2000).astype(int)
        days_from_deadline = np.random.randint(30, 270, size=n)

        edu_map = {'No Formal': 0, 'Lower Secondary': 1, 'Upper Secondary': 2, 'Bachelor': 3, 'Master': 4}
        edu_num = demo['education_level'].map(edu_map).values
        region_effect = np.where(demo['region'].str.contains('Kongo'), -5, 5)
        base_score = 40 + edu_num * 10 + region_effect + np.random.normal(0, 15, size=n)
        score = base_score.clip(0, 100).astype(int)

        return pd.DataFrame({
            'id_student': demo['id_student'],
            'total_clicks': clicks,
            'days_from_deadline': days_from_deadline,
            'score': score
        })

    def generate_kongo_experiment(self, demo):
        """
        Simula un experimento controlado de alfabetización digital en el Kongo.

        Diseño del experimento:
            - Solo estudiantes de regiones Kongo participan
            - Asignación aleatoria 50/50 a grupo tratamiento o control
            - El tratamiento mejora acceso digital, velocidad de internet
              y resultado post-test
            - Grupo tratamiento: +18 puntos promedio en post-test
            - Grupo control: solo +5 puntos promedio (efecto Hawthorne)

        Variables:
            - is_kongo: Indicador de participación en experimento (0/1)
            - treatment: Asignación al grupo tratamiento (0/1)
            - kongo_pre_test: Puntaje pre-intervención (NaN si no participa)
            - kongo_post_test: Puntaje post-intervención (NaN si no participa)
            - digital_access: Nivel de acceso digital (high/medium/low/unknown)
            - internet_speed_mbps: Velocidad de internet (NaN si no participa)

        Args:
            demo: DataFrame demográfico (necesita region)

        Returns:
            pd.DataFrame: Datos del experimento Kongo
        """
        n = self.n
        is_kongo = demo['region'].str.contains('Kongo')
        n_kongo = is_kongo.sum()

        treatment = np.full(n, 0)
        treatment[is_kongo] = np.random.choice([0, 1], size=n_kongo, p=[0.5, 0.5])

        pre_test = np.full(n, np.nan)
        post_test = np.full(n, np.nan)
        digital_access = np.full(n, 'unknown')
        internet_speed = np.full(n, np.nan)

        pre_test[is_kongo] = np.random.normal(45, 12, size=n_kongo).clip(0, 100).astype(int)
        access_vals = np.where(treatment[is_kongo] == 1,
                                np.random.choice(['high', 'medium', 'low'], size=n_kongo, p=[0.4, 0.4, 0.2]),
                                np.random.choice(['high', 'medium', 'low'], size=n_kongo, p=[0.2, 0.3, 0.5]))
        digital_access[is_kongo] = access_vals
        internet_speed[is_kongo] = np.where(treatment[is_kongo] == 1,
                                            np.random.uniform(5, 50, size=n_kongo),
                                            np.random.uniform(1, 20, size=n_kongo)).round(1)
        post_test[is_kongo] = (pre_test[is_kongo] + np.where(treatment[is_kongo] == 1, 18, 5)
                               + np.random.normal(0, 8, size=n_kongo)).clip(0, 100).astype(int)

        return pd.DataFrame({
            'id_student': demo['id_student'],
            'is_kongo': is_kongo.astype(int),
            'treatment': treatment,
            'kongo_pre_test': pre_test,
            'kongo_post_test': post_test,
            'digital_access': digital_access,
            'internet_speed_mbps': internet_speed
        })

    def generate_targets(self, demo, interactions):
        """
        Genera variables target para los tres tipos de modelos.

        Variables:
            - passed (dicotómica): 1 si score >= 40, 0 en otro caso
            - performance_tier (ordinal): Fail (<30), Low (30-55), Medium (55-80), High (>80)
            - final_grade (intervalo/razón): Score normalizado a [0.0, 1.0]

        Args:
            demo: DataFrame demográfico
            interactions: DataFrame de interacciones (necesita score)

        Returns:
            pd.DataFrame: Variables target
        """
        n = self.n
        passed = (interactions['score'] >= 40).astype(int)
        score = interactions['score']
        performance_tier = pd.cut(score, bins=[0, 30, 55, 80, 101],
                                  labels=['Fail', 'Low', 'Medium', 'High'],
                                  ordered=True)
        final_grade = score / 100.0

        return pd.DataFrame({
            'id_student': demo['id_student'],
            'passed': passed,
            'performance_tier': performance_tier,
            'final_grade': final_grade
        })

    def inject_missing(self, df, cols, rate=0.05):
        """
        Inyecta valores faltantes (NaN) en columnas específicas para simular
        datos reales con missing values.

        Args:
            df: DataFrame original
            cols: Lista de columnas en las que inyectar NaN
            rate: Proporción de valores a hacer faltantes

        Returns:
            pd.DataFrame: DataFrame con valores NaN inyectados
        """
        df = df.copy()
        for col in cols:
            if col in df.columns:
                idx = np.random.choice(df.index, size=int(len(df) * rate), replace=False)
                df.loc[idx, col] = np.nan
        return df

    def build_full_dataset(self):
        """
        Construye el dataset completo uniendo todos los componentes:
        demográficos + curso + interacciones + experimento Kongo + targets.

        Inyecta valores faltantes en:
            - score, total_clicks, internet_speed_mbps, kongo_pre_test (5%)
            - studied_credits (2%)

        Returns:
            pd.DataFrame: Dataset completo con 21 columnas y 2000 registros
        """
        demo = self.generate_student_demographics()
        course = self.generate_course_data(demo)
        interactions = self.generate_interaction_data(demo, course)
        kong = self.generate_kongo_experiment(demo)
        targets = self.generate_targets(demo, interactions)

        df = demo.merge(course, on='id_student')
        df = df.merge(interactions, on='id_student')
        df = df.merge(kong, on='id_student')
        df = df.merge(targets, on='id_student')

        df = self.inject_missing(df, ['score', 'total_clicks', 'internet_speed_mbps', 'kongo_pre_test'], rate=0.05)
        df = self.inject_missing(df, ['studied_credits'], rate=0.02)

        return df

    def generate_all_csv(self, output_dir:Path):
        """
        Genera el dataset completo y lo exporta a archivos CSV.

        Archivos generados:
            - oulad_kongo_full.csv: Dataset completo (2000 registros x 21 columnas)
            - features.csv: Solo variables de entrada
            - targets.csv: Solo variables objetivo

        Args:
            output_dir: Directorio de salida (relativo desde src/)
        """
        df = self.build_full_dataset()
        df.to_csv(f'{output_dir}/oulad_kongo_full.csv', index=False)

        features = ['id_student', 'age_band', 'gender', 'region', 'education_level',
                   'code_module', 'num_of_prev_attempts', 'studied_credits',
                   'total_clicks', 'days_from_deadline', 'score',
                   'is_kongo', 'treatment', 'kongo_pre_test', 'kongo_post_test',
                   'digital_access', 'internet_speed_mbps']
        df[features].to_csv(f'{output_dir}/features.csv', index=False)

        targets = ['id_student', 'passed', 'performance_tier', 'final_grade']
        df[targets].to_csv(f'{output_dir}/targets.csv', index=False)

        print(f"Dataset generado: {len(df)} registros.")
        print(f"Columnas: {list(df.columns)}")
        print(f"Tasa de aprobación: {df['passed'].mean():.2%}")
        print(f"Missing en score: {df['score'].isna().sum()}")
        print(f"Missing en kongo_pre_test: {df['kongo_pre_test'].isna().sum()}")
        return df

def main():
    # Punto de entrada: genera los datos sintéticos automáticamente
    src = Path(__file__).parent.parent.parent
    output_dir = src / 'data'
    output_dir.mkdir(exist_ok=True)
    gen = OULADDataGenerator(n_students=2000)
    gen.generate_all_csv(output_dir)