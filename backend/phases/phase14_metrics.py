from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def run(context: dict[str, Any]) -> dict[str, Any]:
    X = context.get("X")
    algorithms = context.get("algorithms", [])
    if X is None or X.empty:
        return {"phase": 14, "name": "Evaluate Metrics", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}
    if len(X) < 2:
        return {"phase": 14, "name": "Evaluate Metrics", "status": "warning", "summary": "At least two rows are required to evaluate clusters.", "warnings": ["Dataset has fewer than two rows."], "errors": [], "tables": {}}
    rows = []
    warnings = []
    sample_size = context.get("algorithm_sample_size", len(X))
    eval_X = X.sample(n=sample_size, random_state=42) if sample_size < len(X) else X
    for algorithm in algorithms:
        model = algorithm["model"]
        try:
            labels = model.fit_predict(eval_X)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"{algorithm['name']} could not be evaluated: {exc}")
            rows.append({"algorithm": algorithm["name"], "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan, "error": str(exc)})
            continue
        algorithm["evaluation_labels"] = labels
        algorithm["evaluation_features"] = eval_X
        valid_rows = labels != -1 if algorithm["name"] == "DBSCAN" else np.ones(len(labels), dtype=bool)
        metric_X = eval_X.loc[valid_rows]
        metric_labels = labels[valid_rows]
        unique_labels = np.unique(metric_labels)
        cluster_count = len(np.unique(labels))
        if len(unique_labels) < 2 or len(unique_labels) >= len(metric_X):
            warnings.append(f"{algorithm['name']} produced {len(unique_labels)} cluster(s); internal validation metrics are undefined.")
            rows.append({"algorithm": algorithm["name"], "clusters": cluster_count, "noise_points": int(np.sum(labels == -1)), "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan, "error": "Metrics undefined for this number of clusters."})
            continue
        try:
            rows.append({
                "algorithm": algorithm["name"],
                "clusters": cluster_count,
                "noise_points": int(np.sum(labels == -1)),
                "silhouette": silhouette_score(metric_X, metric_labels, sample_size=min(1000, len(metric_X)), random_state=42),
                "calinski_harabasz": calinski_harabasz_score(metric_X, metric_labels),
                "davies_bouldin": davies_bouldin_score(metric_X, metric_labels),
                "error": "",
            })
        except ValueError as exc:
            warnings.append(f"{algorithm['name']} metrics could not be calculated: {exc}")
            rows.append({"algorithm": algorithm["name"], "clusters": cluster_count, "noise_points": int(np.sum(labels == -1)), "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan, "error": str(exc)})
    context["metric_table"] = pd.DataFrame(rows)
    summary = "Calculated internal validation metrics to compare candidate clustering solutions."
    valid = context["metric_table"]["silhouette"].notna().any() if rows else False
    status = "completed" if valid else "warning"
    if not valid:
        warnings.append("No candidate produced valid cluster metrics; the default KMeans solution will be used.")
    return {"phase": 14, "name": "Evaluate Metrics", "status": status, "summary": summary, "warnings": warnings, "errors": [], "tables": {"Metrics": pd.DataFrame(rows)}}
