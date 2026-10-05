from __future__ import annotations

from typing import Any

import pandas as pd

from backend.utils.preprocessing import prepare_feature_frame, transform_feature_frame


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 12, "name": "Prepare X", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}
    selected = context.get("selected_features", [])
    preprocessor = context.get("preprocessor")
    if not selected or preprocessor is None:
        return {"phase": 12, "name": "Prepare X", "status": "warning", "summary": "Preprocessing or usable features are missing.", "warnings": ["The feature matrix could not be transformed."], "errors": [], "tables": {}}
    feature_frame = context.get("raw_feature_frame")
    if feature_frame is None:
        feature_frame = prepare_feature_frame(
            df,
            selected,
            context.get("categorical_features", []),
            context.get("source_numeric_columns", []),
        )
    X = transform_feature_frame(feature_frame, preprocessor)
    transformed_features = X.columns.tolist()
    context["X"] = X
    context["transformed_feature_order"] = transformed_features
    summary = f"Prepared the model matrix with {len(transformed_features)} transformed features from {len(selected)} source columns."
    return {"phase": 12, "name": "Prepare X", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Feature order": pd.DataFrame({"feature": transformed_features})}}
