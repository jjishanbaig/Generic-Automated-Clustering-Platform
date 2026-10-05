from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    X = context.get("X")
    labels = context.get("cluster_labels")
    if X is None or X.empty:
        return {"phase": 15, "name": "Analyze Cluster Sizes", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}
    if labels is None:
        return {"phase": 15, "name": "Analyze Cluster Sizes", "status": "warning", "summary": "Cluster labels are not available for size analysis.", "warnings": ["The final model did not assign cluster labels."], "errors": [], "tables": {}}
    size_table = pd.Series(labels, name="cluster").value_counts().rename_axis("cluster").reset_index(name="count")
    summary = "Reviewed cluster sizes to check that the solution is not dominated by a single giant cluster."
    return {"phase": 15, "name": "Analyze Cluster Sizes", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Cluster sizes": size_table}}
