from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    X = context.get("X")
    algorithms = context.get("algorithms", [])
    if X is None or X.empty:
        return {"phase": 17, "name": "Train Final Model", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}
    chosen = context.get("model_choice", {}).get("algorithm", "KMeans")
    if not algorithms:
        return {"phase": 17, "name": "Train Final Model", "status": "warning", "summary": "No clustering algorithm is available.", "warnings": ["No candidate models were prepared."], "errors": [], "tables": {}}
    selected = context.get("selected_algorithm") or next(
        (algorithm for algorithm in algorithms if algorithm["name"] == chosen),
        algorithms[0],
    )
    model = selected["model"]
    labels = selected.get("evaluation_labels")
    reference_features = selected.get("evaluation_features")
    if labels is None or reference_features is None:
        return {"phase": 17, "name": "Train Final Model", "status": "warning", "summary": "The selected model was not fitted during evaluation.", "warnings": ["No fitted candidate is available."], "errors": [], "tables": {}}
    context["model"] = model
    context["model_reference_features"] = reference_features
    context["model_reference_labels"] = labels
    summary = f"Final model selected using {chosen}; it was fitted on {len(reference_features)} evaluation rows."
    return {"phase": 17, "name": "Train Final Model", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Model": pd.DataFrame({"algorithm": [chosen], "fit_rows": [len(reference_features)]})}}
