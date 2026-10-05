from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame()).copy()
    if df.empty:
        return {"phase": 3, "name": "Clean the Data", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    before_rows = len(df)
    warnings = []

    df = df.drop_duplicates().copy()
    if len(df) != before_rows:
        warnings.append(f"Removed {before_rows - len(df)} duplicate rows.")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        else:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().mean() > 0.3:
                warnings.append(f"Column '{col}' has a high missing-value rate and may need review before clustering.")
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"Column '{col}' is constant and may be removed or treated as low-variance.")

    df = df.replace([np.inf, -np.inf], np.nan)
    context["data"] = df
    context["cleaning_summary"] = {
        "rows_after_dedupe": len(df),
        "missing_by_column": df.isna().sum().to_dict(),
    }

    summary = f"Data was cleaned; {before_rows - len(df)} duplicate rows removed and invalid numeric values were coerced to NaN."
    return {"phase": 3, "name": "Clean the Data", "status": "completed", "summary": summary, "warnings": warnings, "errors": [], "tables": {"Cleaning summary": pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values})}}
