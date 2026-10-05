from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd

from backend.phases.phase01_problem import run as phase01
from backend.phases.phase02_data_understanding import run as phase02
from backend.phases.phase03_cleaning import run as phase03
from backend.phases.phase04_clustering_unit import run as phase04
from backend.phases.phase05_categorical import run as phase05
from backend.phases.phase06_feature_engineering import run as phase06
from backend.phases.phase07_feature_validation import run as phase07
from backend.phases.phase08_feature_selection import run as phase08
from backend.phases.phase09_outliers import run as phase09
from backend.phases.phase10_scaling import run as phase10
from backend.phases.phase11_correlation import run as phase11
from backend.phases.phase12_prepare_x import run as phase12
from backend.phases.phase13_algorithms import run as phase13
from backend.phases.phase14_metrics import run as phase14
from backend.phases.phase15_cluster_sizes import run as phase15
from backend.phases.phase16_model_selection import run as phase16
from backend.phases.phase17_final_model import run as phase17
from backend.phases.phase18_assign_clusters import run as phase18
from backend.phases.phase19_cluster_profile import run as phase19
from backend.phases.phase20_visualization import run as phase20
from backend.phases.phase21_interpretation import run as phase21
from backend.phases.phase22_stability import run as phase22
from backend.phases.phase23_save_model import run as phase23
from backend.phases.phase24_prediction import run as phase24
from backend.utils.preprocessing import (
    predict_cluster_labels,
    prepare_feature_frame,
    transform_feature_frame,
)


class GenericClusteringPipeline:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.phase_results: dict[int, dict[str, Any]] = {}
        self.model_bundle: dict[str, Any] = {}
        self.clustered_data = pd.DataFrame()

    def run(self, df: pd.DataFrame) -> dict[int, dict[str, Any]]:
        self.phase_results = {}
        self.model_bundle = {}
        self.clustered_data = pd.DataFrame()
        source_numeric_columns = [
            column
            for column in df.select_dtypes(include=["number"]).columns.tolist()
            if column.lower() not in {"id", "name", "customer_id", "user_id", "email", "uuid"}
            and "_id" not in column.lower()
        ]
        context = {
            "data": df.copy(),
            "input_data": df.copy(),
            "input_columns": df.columns.tolist(),
            "source_numeric_columns": source_numeric_columns,
            "config": self.config,
            "selected_features": [],
            "feature_order": [],
            "X": None,
            "model": None,
            "scaler": None,
            "preprocessor": None,
            "cluster_labels": None,
            "algorithms": [],
            "metric_table": None,
            "metadata": {},
            "results": {},
        }

        phases = [
            phase01, phase02, phase03, phase04, phase05, phase06, phase07,
            phase08, phase09, phase10, phase11, phase12, phase13, phase14,
            phase16, phase17, phase18, phase15, phase19, phase20, phase21,
            phase22, phase23, phase24,
        ]

        for phase in phases:
            result = phase(context)
            context["results"][result["phase"]] = result
            self.phase_results[result["phase"]] = result
            if "model_bundle" in result:
                self.model_bundle.update(result["model_bundle"])
            if "clustered_data" in result:
                self.clustered_data = result["clustered_data"]
        return self.phase_results

    def predict(self, df: pd.DataFrame):
        if "blob" not in self.model_bundle:
            raise ValueError("No trained model bundle is available.")
        buffer = io.BytesIO(self.model_bundle["blob"])
        bundle = joblib.load(buffer)
        feature_order = bundle.get("feature_order", [])
        model = bundle.get("model")
        preprocessor = bundle.get("preprocessor")
        if model is None or preprocessor is None or not feature_order:
            raise ValueError("The saved model bundle is incomplete.")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Prediction input must be a non-empty pandas DataFrame.")
        feature_frame = prepare_feature_frame(
            df,
            feature_order,
            bundle.get("categorical_features", []),
            bundle.get("source_numeric_columns", []),
        )
        transformed_frame = transform_feature_frame(feature_frame, preprocessor)
        return predict_cluster_labels(
            model,
            transformed_frame,
            bundle.get("model_reference_features"),
            bundle.get("model_reference_labels"),
        )
