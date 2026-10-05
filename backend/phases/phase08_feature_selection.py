from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 8, "name": "Select Features", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    excluded = []
    for col in df.columns:
        lower = col.lower()
        if lower in {"id", "name", "customer_id", "user_id", "email", "uuid"} or "_id" in lower:
            excluded.append(col)

    numeric = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded and df[column].nunique(dropna=True) > 1
    ]
    categorical = [
        column for column in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
        if (
            column not in excluded
            and 1 < df[column].nunique(dropna=True) <= context.get("config", {}).get("max_categories", 30)
        )
    ]
    selected = numeric + categorical

    context["selected_features"] = selected
    context["feature_order"] = selected
    context["numeric_features"] = numeric
    context["categorical_features"] = categorical
    context["excluded_features"] = excluded
    summary = (
        f"Selected {len(numeric)} numeric and {len(categorical)} categorical features; "
        f"excluded {len(excluded)} ID/name-like columns."
    )
    return {
        "phase": 8,
        "name": "Select Features",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {
            "Selected features": pd.DataFrame({"feature": selected}),
            "Categorical features": pd.DataFrame({"feature": categorical}),
            "Excluded features": pd.DataFrame({"feature": excluded}),
        },
    }
