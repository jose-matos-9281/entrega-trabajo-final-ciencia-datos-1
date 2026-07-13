"""Read-only production data access for OULAD training artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv


KEY_COLUMNS = ["id_student", "code_module", "code_presentation"]
TARGET_COLUMNS = ["final_result", "passed", "performance_tier", "weighted_assessment_score"]

PROJECT_DIR = Path(__file__).resolve().parents[2]
SQL_PATH = PROJECT_DIR / "sql" / "oulad_training_mart.sql"


def _postgres_dsn() -> str:
    """Load POSTGRES_DSN without overriding an explicitly supplied environment value."""
    load_dotenv(PROJECT_DIR / ".env", override=False)
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is required to generate OULAD training artifacts")
    return dsn


def _quote_sql_literal(value: str) -> str:
    """Quote the DSN for DuckDB ATTACH without ever logging or persisting it."""
    return "'" + value.replace("'", "''") + "'"


def load_training_mart(cutoff_day: int) -> pd.DataFrame:
    """Return a leakage-safe enrollment-level mart extracted from Neon read-only."""
    if isinstance(cutoff_day, bool) or not isinstance(cutoff_day, int):
        raise TypeError("cutoff_day must be an integer")

    dsn = _postgres_dsn()
    sql = SQL_PATH.read_text(encoding="utf-8")
    connection = duckdb.connect()
    try:
        try:
            connection.execute("LOAD postgres")
        except duckdb.Error as error:
            raise RuntimeError(
                "DuckDB's postgres extension must be installed and loadable before "
                "generating OULAD training artifacts. Install it once in the runtime "
                "environment, then retry."
            ) from error
        connection.execute(
            f"ATTACH {_quote_sql_literal(dsn)} AS oulad (TYPE postgres, READ_ONLY)"
        )
        frame = connection.execute(sql, [cutoff_day]).fetchdf()
    finally:
        connection.close()

    return frame.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
