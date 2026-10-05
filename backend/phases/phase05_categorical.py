from __future__ import annotations

from typing import Any

import pandas as pd


def run(context: dict[str, Any]) -> dict[str, Any]:
    df = context.get("data", pd.DataFrame())
    if df.empty:
        return {"phase": 5, "name": "Handle Categorical Data", "status": "warning", "summary": "No data is available for this phase.", "warnings": ["Dataset is empty."], "errors": [], "tables": {}}

    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    high_cardinality = [c for c in cat_cols if df[c].nunique(dropna=True) > 10]
    excluded = [c for c in cat_cols if c.lower() in {"id", "name", "customer_id", "user_id", "email", "uuid"}]
    strategy_map = []
    for col in cat_cols:
        if col in excluded:
            strategy = "exclude"
        elif df[col].nunique(dropna=True) <= 5:
            strategy = "one-hot / ordinal"
        elif df[col].nunique(dropna=True) <= 20:
            strategy = "frequency encoding / target-aware conversion"
        else:
            strategy = "frequency encoding"
        strategy_map.append({"column": col, "unique_values": int(df[col].nunique(dropna=True)), "strategy": strategy})

    context["categorical_columns"] = cat_cols
    context["categorical_strategy_map"] = strategy_map
    summary = (
        f"Detected {len(cat_cols)} categorical fields. "
        f"{len(high_cardinality)} are high-cardinality and will be handled with frequency encoding or exclusion when they are identifiers."
    )
    return {
        "phase": 5,
        "name": "Handle Categorical Data",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {
            "Categorical columns": pd.DataFrame(strategy_map),
            "Excluded IDs": pd.DataFrame({"column": excluded}),
        },
    }
