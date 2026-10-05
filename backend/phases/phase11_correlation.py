from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 11, "name": "Check Correlation / Redundancy", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    features = context.get("numeric_features", df.select_dtypes(include=["number"]).columns.tolist())
    if not features:
        return {"phase": 11, "name": "Check Correlation / Redundancy", "status": "warning", "summary": "No numeric features are available for correlation analysis.", "warnings": ["Categorical features are excluded from numeric correlation checks."], "errors": [], "tables": {}}
    corr = df[features].corr().round(3)
    high_corr = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1:]:
            val = corr.loc[col_a, col_b]
            if abs(val) >= 0.9:
                high_corr.append({"feature_a": col_a, "feature_b": col_b, "correlation": float(val)})

    context["correlation_threshold"] = 0.9
    context["correlation_report"] = high_corr
    summary = f"Correlation checks completed across {len(features)} features and found {len(high_corr)} highly correlated pairs. These pairs are reported without automatically removing them unless the user explicitly chooses to do so."
    return {
        "phase": 11,
        "name": "Check Correlation / Redundancy",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Correlation heatmap": corr, "Highly correlated pairs": pd.DataFrame(high_corr)},
    }
