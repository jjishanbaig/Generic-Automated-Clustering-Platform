from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(
    df: pd.DataFrame,
    source_numeric_columns: Sequence[str] | None = None,
    max_ratio_features: int = 20,
) -> pd.DataFrame:
    working = df.copy()
    for column in source_numeric_columns or []:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    numeric_columns = list(source_numeric_columns) if source_numeric_columns is not None else (
        working.select_dtypes(include=["number"]).columns.tolist()
    )
    numeric_columns = [column for column in numeric_columns if column in working.columns]
    created = 0

    for index, left in enumerate(numeric_columns):
        for right in numeric_columns[index + 1:]:
            if created >= max_ratio_features:
                break
            ratio_name = f"{left}_to_{right}_ratio"
            if ratio_name in working.columns:
                continue
            denominator = working[right].replace(0, np.nan)
            if denominator.notna().any():
                working[ratio_name] = working[left] / denominator
                created += 1
        if created >= max_ratio_features:
            break

    for column in numeric_columns[:3]:
        log_name = f"log_{column}"
        if log_name not in working.columns:
            working[log_name] = np.log1p(working[column].where(working[column] >= 0))

    return working


def prepare_feature_frame(
    df: pd.DataFrame,
    feature_order: Sequence[str],
    categorical_features: Sequence[str],
    source_numeric_columns: Sequence[str],
) -> pd.DataFrame:
    working = engineer_features(df, source_numeric_columns)
    for column in feature_order:
        if column not in working.columns:
            working[column] = np.nan

    features = working[list(feature_order)].copy()
    for column in categorical_features:
        if column in features.columns:
            features[column] = features[column].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )
    return features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def transform_feature_frame(feature_frame: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    transformed = preprocessor.transform(feature_frame)
    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=feature_frame.index,
    )


def predict_cluster_labels(
    model: object,
    features: pd.DataFrame,
    reference_features: pd.DataFrame | None = None,
    reference_labels: np.ndarray | None = None,
) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict(features)
    if reference_features is None or reference_labels is None or len(reference_labels) == 0:
        raise ValueError("This clustering algorithm requires fitted reference labels for new assignments.")

    labels = np.asarray(reference_labels)
    reference = reference_features
    non_noise = labels != -1
    if non_noise.any():
        reference = reference.loc[non_noise]
        labels = labels[non_noise]
    classifier = KNeighborsClassifier(
        n_neighbors=min(5, len(reference)),
        weights="distance",
    )
    classifier.fit(reference, labels)
    return classifier.predict(features)
