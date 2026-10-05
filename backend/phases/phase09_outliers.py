from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 9, "name": "Analyze Outliers", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    features = context.get("numeric_features", df.select_dtypes(include=["number"]).columns.tolist())
    rows = []
    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append({
            "feature": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": out_count,
            "outlier_rate": round(out_count / max(len(df), 1), 4),
        })

    context["outlier_summary"] = rows
    summary = "IQR bounds were used to identify outliers while preserving the option to keep, cap, or remove them based on the clustering use case."
    return {
        "phase": 9,
        "name": "Analyze Outliers",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Outlier summary": pd.DataFrame(rows)},
    }
