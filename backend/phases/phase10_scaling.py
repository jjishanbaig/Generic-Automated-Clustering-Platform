from __future__ import annotations

from typing import Any

import pandas as pd

from backend.utils.preprocessing import build_preprocessor, prepare_feature_frame


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 10, "name": "Scale Features", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    features = context.get("selected_features", [])
    numeric_features = context.get("numeric_features", [])
    categorical_features = context.get("categorical_features", [])
    if not features:
        return {"phase": 10, "name": "Scale Features", "status": "warning", "summary": "No suitable features were selected for clustering.", "warnings": ["Select at least one non-constant numeric or low-cardinality categorical feature."], "errors": [], "tables": {}}

    feature_frame = prepare_feature_frame(
        df,
        features,
        categorical_features,
        context.get("source_numeric_columns", []),
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(feature_frame)
    context["preprocessor"] = preprocessor
    context["raw_feature_frame"] = feature_frame
    context["scaling_strategy"] = "Median imputation + StandardScaler for numeric features; most-frequent imputation + one-hot encoding for categorical features."
    summary = "Fitted reusable preprocessing for numeric and categorical features, including missing-value imputation and scaling/encoding."
    return {
        "phase": 10,
        "name": "Scale Features",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Prepared input sample": feature_frame.head()},
    }
