from __future__ import annotations

from typing import Any

import pandas as pd

from backend.utils.preprocessing import engineer_features


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame()).copy()
    if df.empty:
        return {"phase": 6, "name": "Engineer Useful Features", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    numeric_cols = context.get("source_numeric_columns", df.select_dtypes(include=["number"]).columns.tolist())
    engineered_df = engineer_features(df, numeric_cols)
    engineered = [column for column in engineered_df.columns if column not in df.columns]

    context["data"] = engineered_df
    context["engineered_features"] = engineered
    summary = "Ratio and log transforms were created only where they are meaningful and non-arbitrary for clustering."
    return {
        "phase": 6,
        "name": "Engineer Useful Features",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Engineered columns": pd.DataFrame({"column": engineered})},
    }
