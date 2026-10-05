from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    labels = context.get("cluster_labels")
    if labels is None:
        return {"phase": 21, "name": "Interpret Clusters", "status": "warning", "summary": "Clusters have not been assigned yet.", "warnings": ["Cluster labels are missing."], "errors": [], "tables": {}}
    counts = pd.Series(labels).value_counts().sort_index()
    summary = "The cluster assignments were translated into interpretable groups and size patterns."
    return {"phase": 21, "name": "Interpret Clusters", "status": "completed", "summary": summary + f" Cluster counts: {counts.to_dict()}", "warnings": [], "errors": [], "tables": {"Cluster counts": counts.reset_index().rename(columns={"index": "cluster", 0: "count"})}}
