# Generic Automated Clustering Machine Learning Platform

This project provides a modular Python + Streamlit platform for uploading CSV data and running a 24-phase clustering workflow. It profiles data, selects numeric and low-cardinality categorical inputs, imputes missing values, scales numeric columns, one-hot encodes categorical columns, compares K-Means, Gaussian Mixture, Mean Shift, Hierarchical (Ward) Clustering, and DBSCAN, profiles clusters, and supports interactive predictions.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Using the app

1. Upload a CSV file with at least two observations and one usable feature.
2. Review the workflow, feature diagnostics, model metrics, cluster profiles, and visualizations from the sidebar.
3. Open **Prediction** to enter one observation and receive its cluster assignment.
4. Open **Downloads** to save the fitted model bundle or the input CSV with cluster labels.

Categorical columns with a high number of distinct values are excluded by default to avoid turning identifiers or free text into very wide one-hot encoded matrices. The threshold can be changed by passing `{"max_categories": value}` to `GenericClusteringPipeline`.

Model comparison uses a deterministic sample of up to 1,500 rows by default to keep hierarchical and density-based methods practical on larger uploads; adjust `max_algorithm_samples` in `GenericClusteringPipeline` to change this. Silhouette, Calinski-Harabasz, and Davies-Bouldin scores are calculated on valid non-noise DBSCAN points. The best candidate is selected by silhouette, then Calinski-Harabasz. For Agglomerative Clustering and DBSCAN, which do not natively assign new observations, prediction uses a nearest-neighbor assignment based on fitted reference points.

## Structure

```text
backend/
  pipeline.py
  phases/
  models/
  results/
  utils/
app.py
requirements.txt
uploads/
```
