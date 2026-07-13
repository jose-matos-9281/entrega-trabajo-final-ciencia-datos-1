# OULAD: ejecución end-to-end

El flujo reproducible genera los datos de entrenamiento desde Neon, ejecuta el EDA y entrena el pipeline ML, siempre en ese orden.

## Primera preparación

El script instala [`uv`](https://docs.astral.sh/uv/) sólo si no está disponible y sincroniza las dependencias de `oulad-ml-project`. No requiere `POSTGRES_DSN` ni accede a Neon.

Desde la raíz del repositorio:

```bash
./oulad-ml-project/scripts/setup_oulad_environment.sh
```

Desde `oulad-ml-project/`:

```bash
./scripts/setup_oulad_environment.sh
```

## Ejecutar el flujo completo

Después de preparar el entorno, configurá `POSTGRES_DSN` de forma privada con un DSN de Neon de solo lectura. Hacelo únicamente antes de ejecutar el flujo de extracción; no lo incluyas en el repositorio, comandos ni logs.

```bash
export POSTGRES_DSN='<tu-dsn-privado>'
```

Desde la raíz del repositorio:

```bash
./oulad-ml-project/scripts/run_oulad_pipeline.sh
```

Desde `oulad-ml-project/`, con la variable ya exportada:

```bash
./scripts/run_oulad_pipeline.sh
```

El corte por defecto es el día 30. Para cambiarlo sin modificar archivos:

```bash
CUTOFF_DAY=45 ./oulad-ml-project/scripts/run_oulad_pipeline.sh
```

El script ejecuta exactamente estas etapas:

1. `uv run create-data --cutoff-day <CUTOFF_DAY> --output-dir data`
2. `uv run eda --input-csv data/oulad_training_full.csv --output-dir output/eda`
3. `uv run ml_pipeline`

## Salidas

| Ruta | Contenido |
|---|---|
| `oulad-ml-project/data/` | Mart de entrenamiento, features, targets y metadatos del corte. |
| `oulad-ml-project/output/eda/` | Reporte, CSVs descriptivos y figuras del EDA. |
| `oulad-ml-project/output/models/academic_risk/<uuid>/` | Bundle autocontenido del modelo entrenado y sus métricas. |
| `oulad-ml-project/output/predictions/` | Predicciones generadas durante inferencia. |

## Ejecutar etapas por separado

Desde `oulad-ml-project/`:

```bash
uv run create-data --cutoff-day 30 --output-dir data
uv run eda --input-csv data/oulad_training_full.csv --output-dir output/eda
uv run ml_pipeline
```

Para inferencia, indicá explícitamente el UUID del bundle generado:

```bash
uv run predict-excel --excel oulad-dataset.xlsx --artifacts-dir output/models/academic_risk/<uuid> --cutoff-day 30 --output output/predictions/excel_predictions.csv
```

Los resultados del modelo describen asociaciones predictivas, no relaciones causales. El directorio UUID del modelo debe suministrarse de forma explícita para inferencia; no se selecciona una ejecución automáticamente.
