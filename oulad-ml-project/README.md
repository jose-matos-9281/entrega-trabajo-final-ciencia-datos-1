# OULAD Enrollment Inference

This project trains leakage-safe OULAD enrollment models from the canonical Neon mart and produces `passed` predictions for the local Excel workbook. The Excel file is inference-only; it is never a training source.

## Quick Path

1. Install the locked environment with `uv sync`.
2. Generate canonical artifacts from Neon at one cutoff day.
3. Evaluate risk-model candidates and persist the selected champion bundle.
4. Predict from the Excel workbook using the identical cutoff day.

```bash
uv sync
uv run create-data --cutoff-day 30 --output-dir data
uv run train-inference-model --training-data data/oulad_training_full.csv --training-metadata data/oulad_training_metadata.json --artifacts-dir artifacts
uv run predict-excel --excel oulad-dataset.xlsx --artifacts-dir artifacts --cutoff-day 30 --output output/excel_predictions.csv
```

## Architecture

| Component | Responsibility |
|---|---|
| Neon/Postgres | Canonical OULAD source queried through DuckDB with `READ_ONLY`. |
| `sql/oulad_training_mart.sql` | Builds one enrollment row per `id_student`, `code_module`, and `code_presentation`. |
| `create-data` | Writes deterministic training artifacts and cutoff metadata. |
| `ModelTrainer` | Owns production risk training, artifact reporting, and reproducible charts. Its legacy multi-target methods are exploratory analysis only. |
| `ml_pipeline` / `train-inference-model` | Coordinate data loading and delegate champion publication to `ModelTrainer`. |
| `predict-excel` | Maps approved Excel sheets to the same enrollment feature contract and writes a new prediction CSV. |

The three enrollment key columns are metadata, not model features. In particular, `id_student` is never a feature. `oulad_training_full.csv` is the enrollment-level mart for EDA and model training; `features.csv` excludes all targets and `targets.csv` contains the keys plus all targets.

## Requirements And Setup

Python 3.12+ and `uv` are required. Dependencies, including DuckDB, pandas, scikit-learn, and openpyxl, are locked in `uv.lock`.

```bash
uv sync
```

Create `oulad-ml-project/.env` locally with a real connection string only on your machine:

```dotenv
POSTGRES_DSN=<your-read-only-neon-postgres-dsn>
```

Do not commit `.env`, print the DSN, or pass it on a command line. Artifact generation loads it only to attach Neon through DuckDB in read-only mode.

## Generate Training Artifacts

Choose one integer cutoff day and retain it across artifact generation, model training, and prediction:

```bash
uv run create-data --cutoff-day 30 --output-dir data
```

This writes:

| Output | Contents |
|---|---|
| `data/oulad_training_full.csv` | Enrollment mart for EDA and model training. |
| `data/features.csv` | Enrollment keys and candidate features, excluding every target column. |
| `data/targets.csv` | Enrollment keys and outcome targets, including `final_result`. |
| `data/features.csv` | Keys plus permitted source features. |
| `data/targets.csv` | Keys plus `passed`, `performance_tier`, and `weighted_assessment_score`. |
| `data/oulad_training_metadata.json` | Artifact version, source, cutoff, and column counts. |

## Train And Persist Inference Artifacts

```bash
uv run train-inference-model --training-data data/oulad_training_full.csv --training-metadata data/oulad_training_metadata.json --artifacts-dir artifacts
```

The command performs a grouped split by `id_student` before fitting any imputer, encoder, or scaler. It evaluates all candidates with `passed=0` as the risk class and selects the champion by `risk_recall`; risk precision, risk F1, ROC-AUC, and general metrics remain guardrails rather than a substitute for the operational objective.

It persists a versioned champion bundle, JSON report, candidate-metrics CSV, and holdout-predictions CSV, plus a stable `artifacts/inference_manifest.json`. It also writes non-interactive PNG charts under `artifacts/figures/`: candidate metric comparison, champion holdout confusion matrix, and feature importance when the champion exposes coefficients or `feature_importances_`. The report references every artifact, records the grouped split, train-only CV selection, holdout metrics, feature contract, and limitations. The manifest repeats the artifact references for discovery while inference continues to consume only the manifest and champion bundle.

The serialized bundle is the evaluated champion, fit only on the training partition. Candidate selection uses GroupKFold only within that partition, prioritizing recall for risk class `passed=0`; the independent grouped holdout is evaluated only after selection. Feature importance is omitted, with an explicit reason in the report, for models that do not expose a supported importance interface. Prediction refuses an artifact whose cutoff or feature contract differs from the requested run.

## Predict From Excel

```bash
uv run predict-excel --excel oulad-dataset.xlsx --artifacts-dir artifacts --cutoff-day 30 --output output/excel_predictions.csv
```

The source workbook remains unchanged. The output contains the three enrollment keys, cutoff, model target, class prediction, probability, `oov_categorical_fields`, and `oov_key_fields`. A sibling `<output>.inference_report.json` records the champion, contract, row count, OOV counts, missing values passed to imputers, probability distribution, and drift warnings. `none` in an OOV field means no value was unseen in training. Unknown modules and presentations remain metadata, never model features, and are reported in `oov_key_fields`; unknown regions and other categorical feature values use the persisted encoder with unknown categories ignored.

The adapter reads only `StudentInfo`, `Registration`, `VLE_clickStream`, and `cursos`. It intentionally does not load assessment sheets, so `score`, `final_result`, `date_submitted`, assessment plans, and other outcomes cannot enter inference.

## Cutoff And Leakage Protection

The training mart includes enrollments registered on or before `cutoff_day` and aggregates VLE events where `date <= cutoff_day`. The Excel adapter applies the same rule.

- Assessment scores and final outcomes are targets or post-cutoff information and are excluded.
- The grouped split by `id_student` occurs before preprocessing; do not replace it with a row split or pre-split feature transformation.
- The adapter rejects missing required fields, duplicate enrollment keys, unmatched enrollment coverage, invalid numeric values, negative clicks, post-cutoff registrations, ambiguous blank VLE modules, and VLE activity without a known enrollment.
- A blank VLE module is resolved only when student and presentation identify exactly one enrollment. This is a deterministic relational join, not imputation.

## Tests

Run the unit suite without Neon credentials:

```bash
uv run python -m unittest
```

Run the opt-in Neon contract test only with a locally configured read-only DSN:

```bash
RUN_NEON_INTEGRATION=1 uv run python -m unittest discover -s tests -p test_training_artifacts.py
```

## Limitations And Troubleshooting

The local Excel contains a different population and normalized sheet/column names. It is valid for inference only, not for fitting, scoring, or replacing the 32k-enrollment Neon training mart. Missing optional raw features are handled by the persisted training-only imputers. New modules, presentations, regions, or other categorical values can cause domain shift even though encoder handling prevents runtime failures; inspect the inference report rather than treating a successful prediction as evidence of quality.

| Symptom | Resolution |
|---|---|
| `Inference artifacts are incomplete` | Run `train-inference-model` before `predict-excel`. |
| Cutoff mismatch | Regenerate/retrain with the intended cutoff, then pass that same value to prediction. |
| Required sheet or column missing | Export the supported four-sheet Excel schema without renaming its required fields. |
| Unresolved VLE module or activity coverage error | Correct the source relation; the adapter will not guess ambiguous enrollment data. |
| DuckDB PostgreSQL extension unavailable | Install/load the extension in the local runtime, then retry `create-data`; no database write is needed. |
