#!/usr/bin/env python3
"""Preview and materialise the eICU feature table expected by AIClinician_core_160219.m."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


try:
    import numpy as np
    import pandas as pd
    from scipy.io import savemat
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
except ImportError as exc:  # pragma: no cover - raised during CLI execution
    OPTIONAL_IMPORT_ERROR = exc
    np = pd = None  # type: ignore
    savemat = create_engine = None  # type: ignore
    Engine = object  # type: ignore
else:
    OPTIONAL_IMPORT_ERROR = None

# Columns listed in AIClinician_core_160219.m:80-95
EXPECTED_FEATURE_COLUMNS = [
    "gender",
    "mechvent",
    "max_dose_vaso",
    "re_admission",
    "age",
    "admissionweight",
    "gcs",
    "hr",
    "sysbp",
    "meanbp",
    "diabp",
    "rr",
    "temp_c",
    "fio2both",
    "potassium",
    "sodium",
    "chloride",
    "glucose",
    "magnesium",
    "calcium",
    "hb",
    "wbc_count",
    "platelets_count",
    "ptt",
    "pt",
    "arterial_ph",
    "pao2",
    "paco2",
    "arterial_be",
    "hco3",
    "arterial_lactate",
    "sofa",
    "sirs",
    "shock_index",
    "pao2_fio2",
    "cumulated_balance_tev",
    "spo2",
    "bun",
    "creatinine",
    "sgot",
    "sgpt",
    "total_bili",
    "inr",
    "input_total_tev",
    "input_4hourly_tev",
    "output_total",
    "output_4hourly",
]

# Columns referenced later in AIClinician_core_160219.m:425-427
META_COLUMNS = ["patientunitstayid", "bloc", "hospmortality"]


DEFAULT_SOURCE = "ai_clinician.eicu_hourly_features"


# fixed Postgres connection details per workflow request
DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "pass"
DB_NAME = "postgres"


def require_dependencies() -> None:
    """Ensure optional dependencies are available before execution."""
    if OPTIONAL_IMPORT_ERROR is not None:
        raise ImportError(
            "Missing optional dependencies. Install with 'pip install pandas sqlalchemy psycopg2-binary scipy'."
        ) from OPTIONAL_IMPORT_ERROR



DEFAULT_FEATURE_QUERY = """
SELECT
    patientunitstayid,
    bloc,
    gender,
    mechvent,
    max_dose_vaso,
    re_admission,
    age,
    admissionweight,
    gcs,
    hr,
    sysbp,
    meanbp,
    diabp,
    rr,
    temp_c,
    fio2both,
    potassium,
    sodium,
    chloride,
    glucose,
    magnesium,
    calcium,
    hb,
    wbc_count,
    platelets_count,
    ptt,
    pt,
    arterial_ph,
    pao2,
    paco2,
    arterial_be,
    hco3,
    arterial_lactate,
    sofa,
    sirs,
    shock_index,
    pao2_fio2,
    cumulated_balance_tev,
    spo2,
    bun,
    creatinine,
    sgot,
    sgpt,
    total_bili,
    inr,
    input_total_tev,
    input_4hourly_tev,
    output_total,
    output_4hourly,
    hospmortality
FROM {source}
{where_clause}
ORDER BY patientunitstayid, bloc
"""

def build_engine() -> Engine:
    """Create a SQLAlchemy engine using the fixed connection details."""
    require_dependencies()
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        future=True,
    )

def preview_tables(engine: Engine, schema: str, limit: int, table: Optional[str]) -> None:
    """Print tables within a schema and optionally preview a sample."""
    require_dependencies()
    list_sql = (
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %(schema)s "
        "UNION "
        "SELECT matviewname FROM pg_matviews WHERE schemaname = %(schema)s "
        "ORDER BY 1"
    )
    tables = pd.read_sql_query(list_sql, engine, params={"schema": schema})
    if tables.empty:
        print(f"No tables found in schema '{schema}'.")
    else:
        print(f"Tables in schema '{schema}':")
        print(tables.to_string(index=False))

    if table:
        preview_sql = f'SELECT * FROM "{schema}"."{table}" LIMIT {limit};'
        sample = pd.read_sql_query(preview_sql, engine)
        print("\nSample rows:")
        print(sample)

def read_feature_frame(
    engine: Engine,
    query: str,
    expected_cols: Iterable[str],
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """Execute query and ensure the result contains the required columns."""
    require_dependencies()
    result = pd.read_sql_query(query, engine, chunksize=chunksize)
    if isinstance(result, pd.DataFrame):
        df = result
    else:
        df = pd.concat(result, ignore_index=True)

    required = list(expected_cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from query output: {missing}")
    return df[required]

def save_as_mat(df: pd.DataFrame, mat_path: Path) -> None:
    """Write MATLAB-friendly arrays (data + column names)."""
    require_dependencies()
    clean = df.copy()
    clean = clean.replace({pd.NA: np.nan, None: np.nan})

    for col in clean.columns:
        series = clean[col]
        if pd.api.types.is_bool_dtype(series):
            clean[col] = series.astype(float)
        elif pd.api.types.is_integer_dtype(series):
            clean[col] = series.astype(float)
        elif pd.api.types.is_object_dtype(series):
            clean[col] = pd.to_numeric(series, errors="coerce")

    data = clean.to_numpy(dtype=float)
    columns = np.array(clean.columns, dtype=object)
    savemat(mat_path, {
        "eICU_data": data,
        "eICU_columns": columns,
    })

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_preview = subparsers.add_parser(
        "preview",
        help="List tables within a schema and optionally show sample rows",
    )
    parser_preview.add_argument(
        "--schema",
        default="ai_clinician",
        help="Schema to inspect (default: %(default)s)",
    )
    parser_preview.add_argument(
        "--table",
        help="Optional table/view to preview",
    )
    parser_preview.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Row count for preview output (default: %(default)s)",
    )

    parser_build = subparsers.add_parser(
        "build",
        help="Extract the eICU features into CSV and optional MATLAB MAT file",
    )
    parser_build.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Qualified table/view supplying the feature columns",
    )
    parser_build.add_argument(
        "--sql-file",
        type=Path,
        help="Path to custom SQL returning the required columns",
    )
    parser_build.add_argument(
        "--where",
        help="Optional SQL WHERE clause (without the keyword)",
    )
    parser_build.add_argument(
        "--chunksize",
        type=int,
        help="Optional chunk size for streaming fetch",
    )
    parser_build.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV path (relative paths recommended)",
    )
    parser_build.add_argument(
        "--output-mat",
        type=Path,
        help="Optional path for MATLAB MAT output",
    )
    parser_build.add_argument(
        "--echo",
        action="store_true",
        help="Echo final SQL prior to execution",
    )

    return parser.parse_args(argv)

def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    engine = build_engine()

    if args.command == "preview":
        preview_tables(engine, args.schema, args.limit, args.table)
        return

    if args.sql_file:
        query = args.sql_file.read_text(encoding="utf-8")
    else:
        where_clause = f"WHERE {args.where}" if args.where else ""
        query = DEFAULT_FEATURE_QUERY.format(source=args.source, where_clause=where_clause)

    if args.echo:
        print("Executing SQL query:\n", query, file=sys.stderr)

    required_cols = META_COLUMNS + EXPECTED_FEATURE_COLUMNS
    frame = read_feature_frame(engine, query, required_cols, args.chunksize)

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"Wrote CSV with {len(frame)} rows to {output_csv}")

    if args.output_mat:
        output_mat = args.output_mat
        output_mat.parent.mkdir(parents=True, exist_ok=True)
        save_as_mat(frame, output_mat)
        print(f"Wrote MATLAB MAT file to {output_mat}")

if __name__ == "__main__":
    main()
