from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 2, "name": "Understand the Data", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    dtype_table = df.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"})
    missing = df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"})
    duplicates = int(df.duplicated().sum())
    unique_counts = df.nunique(dropna=True).reset_index().rename(columns={"index": "column", 0: "unique_count"})
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    text_cols = [c for c in df.columns if df[c].dtype == object and c not in categorical_cols]

    summary = (
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"Numeric columns: {len(numeric_cols)}, categorical columns: {len(categorical_cols)}, "
        f"text-like columns: {len(text_cols)}, duplicates: {duplicates}."
    )
    context["dataset_profile"] = {
        "shape": df.shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "duplicate_rows": duplicates,
    }
    return {
        "phase": 2,
        "name": "Understand the Data",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {
            "Data types": dtype_table.head(20),
            "Missing values": missing.head(20),
            "Unique counts": unique_counts.head(20),
        },
    }
