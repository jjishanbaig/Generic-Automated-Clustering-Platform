from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    clustered = context.get("clustered_data")
    if clustered is None or clustered.empty:
        return {"phase": 20, "name": "Visualize Clusters", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}
    sample = clustered.copy()
    summary = "Cluster labels and transformed features are ready for interactive charting."
    return {"phase": 20, "name": "Visualize Clusters", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Visualization sample": sample.head(100)}}
