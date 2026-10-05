from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    feature_order = context.get("feature_order", [])
    if not feature_order:
        return {"phase": 24, "name": "Build Prediction Pipeline", "status": "warning", "summary": "The prediction pipeline cannot be built without a trained model or feature order.", "warnings": ["Feature order is missing."], "errors": [], "tables": {}}
    summary = f"Prediction pipeline is ready using the stored feature order: {feature_order}."
    return {"phase": 24, "name": "Build Prediction Pipeline", "status": "completed", "summary": summary, "warnings": [], "errors": [], "tables": {"Feature order": pd.DataFrame({"feature": feature_order})}}
