from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    metric_table = context.get("metric_table")
    if metric_table is None or metric_table.empty:
        return {"phase": 16, "name": "Select Algorithm + Parameters", "status": "warning", "summary": "Metrics are missing for model selection.", "warnings": ["No metric table available."], "errors": [], "tables": {}}
    valid = metric_table.dropna(subset=["silhouette"])
    if valid.empty:
        choice = {"algorithm": "KMeans", "basis": "fallback because no candidate had valid internal metrics"}
        status = "warning"
        warnings = ["Selected the default KMeans solution because no candidate had valid metrics."]
    else:
        best_row = valid.sort_values(
            ["silhouette", "calinski_harabasz"], ascending=[False, False]
        ).iloc[0]
        choice = {"algorithm": best_row["algorithm"], "basis": "highest silhouette and Calinski-Harabasz scores"}
        status = "completed"
        warnings = []
    context["model_choice"] = choice
    selected_algorithm = next(
        (algorithm for algorithm in context.get("algorithms", []) if algorithm["name"] == choice["algorithm"]),
        None,
    )
    if selected_algorithm is not None:
        context["selected_algorithm"] = selected_algorithm
    summary = f"Selected {choice['algorithm']} using {choice['basis']}."
    return {"phase": 16, "name": "Select Algorithm + Parameters", "status": status, "summary": summary, "warnings": warnings, "errors": [], "tables": {"Model selection": pd.DataFrame([choice])}}
