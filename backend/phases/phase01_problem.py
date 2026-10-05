from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 1, "name": "Understand the Problem", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}
    summary = f"The clustering objective is defined as an unsupervised problem over {df.shape[0]} rows and {df.shape[1]} columns."
    return {"phase": 1, "name": "Understand the Problem", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {}}
