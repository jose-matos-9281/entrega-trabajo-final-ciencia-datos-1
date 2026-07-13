#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CUTOFF_DAY="${CUTOFF_DAY:-30}"

if ! command -v uv >/dev/null 2>&1; then
    printf 'Error: uv no esta instalado o no se encuentra en PATH.\n' >&2
    exit 1
fi

if [[ -z "${POSTGRES_DSN:-}" ]]; then
    printf 'Error: POSTGRES_DSN debe estar configurada de forma privada antes de ejecutar el pipeline.\n' >&2
    exit 1
fi

if [[ ! "${CUTOFF_DAY}" =~ ^[0-9]+$ ]]; then
    printf 'Error: CUTOFF_DAY debe ser un entero no negativo.\n' >&2
    exit 1
fi

cd "${PROJECT_DIR}"

printf '1/3 Generando datos de entrenamiento desde Neon (cutoff day: %s)...\n' "${CUTOFF_DAY}"
uv run create-data --cutoff-day "${CUTOFF_DAY}" --output-dir data
printf 'Datos disponibles en data/.\n'

printf '2/3 Ejecutando EDA...\n'
uv run eda --input-csv data/oulad_training_full.csv --output-dir output/eda
printf 'Artefactos de EDA disponibles en output/eda/.\n'

printf '3/3 Entrenando el pipeline ML...\n'
uv run ml_pipeline
printf 'Bundle del modelo disponible en output/models/academic_risk/<uuid>/.\n'
