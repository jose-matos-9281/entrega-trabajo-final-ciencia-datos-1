# Hipótesis principal de investigación

## Alerta temprana de resultado académico adverso

### Pregunta de investigación

¿Las señales disponibles hasta el día 30 de una presentación permiten identificar a estudiantes con alto riesgo de terminar en un resultado académico adverso, para priorizar apoyo antes de que abandonen o fracasen?

### Hipótesis

Un modelo de aprendizaje automático entrenado únicamente con información disponible hasta el día 30 puede identificar, con mejor desempeño que un baseline de prevalencia, a las matrículas con riesgo de terminar en `Withdrawn` o `Fail`.

### Resultado a predecir

Se define la variable objetivo binaria `academic_risk` de la siguiente forma:

- `academic_risk = 1` si `final_result` es `Withdrawn` o `Fail`.
- `academic_risk = 0` si `final_result` es `Pass` o `Distinction`.

Esta definición unifica abandono y fracaso como resultados que requieren una intervención académica temprana, sin afirmar que representen el mismo proceso causal.

### Población de estudio

Todas las matrículas activas al día 30 de cada presentación académica.

### Variables predictoras permitidas

El modelo puede usar únicamente variables observables hasta el día 30:

- Información de inscripción y duración de la presentación.
- Número de intentos previos y créditos cursados.
- Actividad acumulada en el VLE: clics, eventos, días activos y sitios visitados.
- Entregas, tipos de actividad y calificaciones registradas antes o en el día 30.

### Prevención de fuga de información

Se excluyen del entrenamiento:

- `final_result`, `academic_risk`, puntuaciones finales y cualquier outcome derivado.
- Identificadores del estudiante.
- Fecha de baja futura.
- Eventos, entregas, calificaciones o agregados generados después del día 30.
- Inscripciones posteriores al día 30.

Las variables de género, región, banda IMD y discapacidad no se emplean como predictores. Se conservan exclusivamente para la auditoría de equidad.

### Diseño de evaluación

- Entrenamiento con presentaciones históricas anteriores.
- Holdout temporal de la presentación `2014J`.
- Validación interna con `GroupKFold` por `id_student`, para evitar que un mismo estudiante aparezca en entrenamiento y validación.
- Todo preprocesamiento, imputación, codificación y ajuste de hiperparámetros se ajusta únicamente con los datos de entrenamiento de cada partición.

### Baseline y métricas

El baseline será un predictor constante basado en la prevalencia de `academic_risk` en entrenamiento.

El modelo se evaluará mediante:

- PR-AUC como métrica principal.
- ROC-AUC.
- Brier score y curva de calibración.
- Precisión y recall dentro del 20% de matrículas con mayor riesgo estimado.

### Criterio de aceptación

La hipótesis se considerará respaldada si, en el holdout temporal:

1. El modelo supera al baseline en PR-AUC.
2. La calibración no es peor que la del baseline.
3. El 20% de matrículas priorizadas concentra una proporción de resultados adversos claramente superior a la selección aleatoria equivalente.

### Acción esperada

El modelo se utilizaría para priorizar tutoría proactiva, contacto temprano, revisión de progreso y planes de recuperación o retención. No se utilizará para sancionar, excluir ni negar recursos a estudiantes.

### Límites e implicaciones éticas

El modelo estima riesgo predictivo; no demuestra que la actividad en el VLE, ni una intervención posterior, causen la reducción del abandono o del fracaso. La efectividad de las intervenciones debe probarse mediante evaluación prospectiva.

Además, se deben auditar la calibración, la tasa de selección, los falsos positivos, los falsos negativos y el beneficio potencial por género, discapacidad, región y banda IMD.
