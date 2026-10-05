from __future__ import annotations

from typing import Any

import pandas as pd

from backend.utils.preprocessing import predict_cluster_labels, prepare_feature_frame, transform_feature_frame


def run(context: dict[str, Any]) -> dict[str, Any]:
    model = context.get("model")
    preprocessor = context.get("preprocessor")
    feature_order = context.get("feature_order", [])
    source_data = context.get("input_data", context.get("data", pd.DataFrame())).copy()
    if source_data.empty or model is None or preprocessor is None or not feature_order:
        return {"phase": 18, "name": "Assign Clusters", "status": "warning", "summary": "Model training is incomplete.", "warnings": ["The final model is not available."], "errors": [], "tables": {}}
    feature_frame = prepare_feature_frame(
        source_data,
        feature_order,
        context.get("categorical_features", []),
        context.get("source_numeric_columns", []),
    )
    transformed = transform_feature_frame(feature_frame, preprocessor)
    labels = predict_cluster_labels(
        model,
        transformed,
        context.get("model_reference_features"),
        context.get("model_reference_labels"),
    )
    context["cluster_labels"] = labels
    source_data["cluster"] = labels
    context["clustered_data"] = source_data
    summary = f"Assigned cluster labels to {len(labels)} rows."
    counts = pd.Series(labels, name="cluster").value_counts().rename_axis("cluster").reset_index(name="count")
    return {"phase": 18, "name": "Assign Clusters", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Cluster counts": counts, "Assignment preview": source_data.head(20)}, "clustered_data": source_data}
