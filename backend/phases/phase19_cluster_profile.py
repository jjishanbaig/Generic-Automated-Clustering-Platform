from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    source_data = context.get("input_data", context.get("data"))
    labels = context.get("cluster_labels")
    if source_data is None or source_data.empty:
        return {"phase": 19, "name": "Profile Clusters", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}
    if labels is None:
        return {"phase": 19, "name": "Profile Clusters", "status": "warning", "summary": "Cluster labels are not available for profiling.", "warnings": ["The final model did not assign cluster labels."], "errors": [], "tables": {}}
    profile_data = source_data.copy()
    profile_data["cluster"] = labels
    numeric = context.get("source_numeric_columns", [])
    numeric = [column for column in numeric if column in profile_data.columns]
    categorical = [column for column in context.get("categorical_features", []) if column in profile_data.columns]
    numeric_profile = profile_data.groupby("cluster")[numeric].mean().round(3) if numeric else pd.DataFrame()
    categorical_profile = profile_data.groupby("cluster")[categorical].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else None
    ) if categorical else pd.DataFrame()
    summary = "Cluster profiles summarize raw numeric feature means and categorical modes for each group."
    return {"phase": 19, "name": "Profile Clusters", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Numeric profile": numeric_profile, "Categorical profile": categorical_profile}}
