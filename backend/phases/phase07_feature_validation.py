from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 7, "name": "Validate Features", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    issues = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col]
        if s.isna().any():
            issues.append({"column": col, "issue_type": "missing_values"})
        if np.isinf(s.to_numpy()).any():
            issues.append({"column": col, "issue_type": "infinite_values"})
        if s.nunique(dropna=True) <= 1:
            issues.append({"column": col, "issue_type": "constant_column"})
        if s.std(ddof=0) == 0:
            issues.append({"column": col, "issue_type": "zero_variance"})
        if not np.isfinite(s.to_numpy()).all():
            issues.append({"column": col, "issue_type": "non_finite"})

    issue_df = pd.DataFrame(issues)
    context["feature_issues"] = issue_df
    summary = f"Reviewed the feature set and found {len(issue_df)} potential issues that should be reviewed before clustering."
    return {"phase": 7, "name": "Validate Features", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Feature issues": issue_df if not issue_df.empty else pd.DataFrame({"column": [], "issue_type": []})}}
