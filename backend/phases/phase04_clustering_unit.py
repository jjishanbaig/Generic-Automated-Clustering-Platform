from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 4, "name": "Identify Clustering Unit", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    candidates = []
    for col in df.columns:
        if col.lower() in {"id", "name", "customer_id", "user_id", "email", "uuid"}:
            continue
        unique_ratio = df[col].nunique(dropna=True) / max(1, len(df))
        if unique_ratio < 0.5 and df[col].notna().any():
            candidates.append(col)

    choice = candidates[0] if candidates else None
    context["clustering_unit_candidates"] = candidates
    context["clustering_unit"] = choice
    if choice:
        summary = f"The dataset suggests a natural clustering unit: '{choice}'. Aggregation may be appropriate if repeated observations belong to the same entity."
    else:
        summary = "The dataset does not clearly suggest a repeated observation structure, so row-level clustering remains acceptable."

    return {"phase": 4, "name": "Identify Clustering Unit", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Candidate columns": pd.DataFrame({"column": candidates})}}
