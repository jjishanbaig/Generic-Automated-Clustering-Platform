from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture


def run(context: dict[str, Any]) -> dict[str, Any]:
    X = context.get("X")
    if X is None or X.empty:
        return {"phase": 13, "name": "Run Multiple Clustering Algorithms", "status": "warning", "summary": "No X matrix is available for this phase.", "warnings": ["X matrix not prepared."], "errors": [], "tables": {}}

    if len(X) < 2:
        return {"phase": 13, "name": "Run Multiple Clustering Algorithms", "status": "warning", "summary": "At least two rows are required for clustering.", "warnings": ["Dataset has fewer than two rows."], "errors": [], "tables": {}}
    n_clusters = min(5, max(2, round(len(X) ** 0.5)))
    sample_size = min(len(X), int(context.get("config", {}).get("max_algorithm_samples", 1500)))
    bandwidth_sample = X.sample(n=min(len(X), 500), random_state=42)
    bandwidth = estimate_bandwidth(
        bandwidth_sample,
        quantile=0.2,
        n_samples=len(bandwidth_sample),
    )
    if not pd.notna(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    algorithms = [
        {"name": "KMeans", "model": KMeans(n_clusters=n_clusters, random_state=42, n_init=10)},
        {"name": "GaussianMixture", "model": GaussianMixture(n_components=n_clusters, random_state=42)},
        {"name": "MeanShift", "model": MeanShift(bandwidth=bandwidth, bin_seeding=True)},
        {"name": "Hierarchical Clustering", "model": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")},
        {"name": "DBSCAN", "model": DBSCAN(eps=0.7, min_samples=5)},
    ]
    context["algorithms"] = algorithms
    context["algorithm_sample_size"] = sample_size
    summary = (
        f"Prepared {len(algorithms)} clustering algorithms for comparison. "
        f"Evaluation is capped at {sample_size} rows for runtime; MeanShift uses an estimated bandwidth of {bandwidth:.3g}."
    )
    return {
        "phase": 13,
        "name": "Run Multiple Clustering Algorithms",
        "status": "completed",
        "summary": summary,
        "warnings": [],
        "errors": [],
        "tables": {"Algorithms": pd.DataFrame({"algorithm": [a["name"] for a in algorithms]})},
    }
