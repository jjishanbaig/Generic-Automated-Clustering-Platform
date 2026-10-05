from __future__ import annotations

from typing import Any

import io

import joblib
import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    model = context.get("model")
    preprocessor = context.get("preprocessor")
    feature_order = context.get("feature_order", [])
    if model is None:
        return {"phase": 23, "name": "Save Model + Scaler + Features", "status": "warning", "summary": "Training is incomplete and no model bundle can be saved.", "warnings": ["Final model is missing."], "errors": [], "tables": {}}
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_order": feature_order,
        "categorical_features": context.get("categorical_features", []),
        "source_numeric_columns": context.get("source_numeric_columns", []),
        "input_columns": list(dict.fromkeys(
            context.get("source_numeric_columns", [])
            + [
                column
                for column in context.get("categorical_features", [])
                if column in context.get("input_columns", [])
            ]
        )),
        "model_reference_features": context.get("model_reference_features"),
        "model_reference_labels": context.get("model_reference_labels"),
        "metadata": context.get("metadata", {}),
    }
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    blob = buffer.getvalue()
    context["model_bundle"] = {"blob": blob, "feature_names": feature_order}
    summary = "Saved the trained model, preprocessing pipeline, and feature metadata for future prediction."
    return {
        "phase": 23,
        "name": "Save Model + Scaler + Features",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Saved artifacts": pd.DataFrame({"artifact": ["model", "preprocessor", "feature_order"], "value": ["trained model", str(preprocessor), str(feature_order)]})},
        "model_bundle": {
            "blob": blob,
            "feature_names": feature_order,
            "input_columns": bundle["input_columns"],
            "categorical_features": bundle["categorical_features"],
        },
    }
