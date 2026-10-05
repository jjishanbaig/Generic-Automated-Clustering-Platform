from __future__ import annotations

import plotly.express as px
import pandas as pd


def missing_values_chart(df: pd.DataFrame):
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if missing.empty:
        return None
    return px.bar(missing, x="column", y="missing_count", title="Missing values by column", color="missing_count")


def corr_heatmap_table(corr: pd.DataFrame):
    if corr is None or corr.empty:
        return None
    return px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")


def cluster_size_chart(labels: pd.Series):
    if labels is None or labels.empty:
        return None
    counts = labels.value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    return px.bar(counts, x="cluster", y="count", title="Cluster size distribution")


def feature_relation_scatter(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str | None = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    plot_df = df[[x_col, y_col]].copy()
    if cluster_col and cluster_col in df.columns:
        plot_df["cluster"] = df[cluster_col]
        return px.scatter(plot_df, x=x_col, y=y_col, color="cluster", title=f"{x_col} vs {y_col}")
    return px.scatter(plot_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
