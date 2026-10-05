from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.pipeline import GenericClusteringPipeline
from backend.utils.csv_reader import read_csv_bytes


st.set_page_config(page_title="Clustering Platform", page_icon="🔎", layout="wide")
SIDEBAR_OPTIONS = [
    "Dashboard",
    "Data Understanding",
    "Data Cleaning",
    "Feature Engineering",
    "Feature Analysis",
    "Algorithm Comparison",
    "Final Clusters",
    "Cluster Profiles",
    "Visualizations",
    "Prediction",
    "Downloads",
    "About",
]


@st.cache_data
def read_csv(contents: bytes) -> pd.DataFrame:
    return read_csv_bytes(contents)


def render_phase(result: dict) -> None:
    st.subheader(f"Phase {result['phase']}: {result['name']}")
    st.write(result.get("summary", ""))
    status = result.get("status", "unknown")
    if status == "completed":
        st.success("Completed")
    elif status == "warning":
        st.warning("Completed with warnings")
    else:
        st.info(f"Status: {status}")
    for warning in result.get("warnings", []):
        st.warning(warning)
    for error in result.get("errors", []):
        st.error(error)
    for name, table in (result.get("tables") or {}).items():
        if table is not None and not table.empty:
            st.markdown(f"#### {name}")
            st.dataframe(table, use_container_width=True)


def render_prediction(pipeline: GenericClusteringPipeline, df: pd.DataFrame) -> None:
    if not pipeline.model_bundle.get("blob"):
        st.info("A trained model is not available for prediction. Check that the dataset has usable features.")
        return

    columns = pipeline.model_bundle.get("input_columns", [])
    categorical = set(pipeline.model_bundle.get("categorical_features", []))
    if not columns:
        st.warning("The saved model does not have input feature metadata.")
        return

    st.write("Enter one observation using the same source columns as the uploaded dataset.")
    values: dict[str, object] = {}
    with st.form("cluster_prediction"):
        widget_columns = st.columns(2)
        for index, name in enumerate(columns):
            widget = widget_columns[index % len(widget_columns)]
            if name in categorical:
                available = df[name].dropna() if name in df.columns else pd.Series(dtype="object")
                mode = available.mode().iloc[0] if not available.mode().empty else ""
                values[name] = widget.text_input(name, value=str(mode), key=f"prediction_{index}")
            else:
                numeric = pd.to_numeric(df[name], errors="coerce").dropna() if name in df.columns else pd.Series(dtype="float64")
                default = float(numeric.median()) if not numeric.empty else 0.0
                values[name] = widget.number_input(
                    name,
                    value=default,
                    format="%.6g",
                    key=f"prediction_{index}",
                )
        submitted = st.form_submit_button("Predict cluster", type="primary")

    if submitted:
        try:
            prediction = pipeline.predict(pd.DataFrame([values]))
        except (ValueError, KeyError) as exc:
            st.error(f"Prediction could not be completed: {exc}")
            return
        st.success(f"Predicted cluster: {prediction[0]}")


def render_visualizations(pipeline: GenericClusteringPipeline) -> None:
    clustered = pipeline.clustered_data
    if clustered.empty or "cluster" not in clustered.columns:
        st.info("Cluster labels are not available to visualize.")
        return

    left, right = st.columns(2)
    counts = clustered["cluster"].value_counts().rename_axis("cluster").reset_index(name="rows")
    left.plotly_chart(
        px.bar(counts, x="cluster", y="rows", color="cluster", title="Cluster sizes"),
        use_container_width=True,
    )

    numeric = [
        column
        for column in pipeline.model_bundle.get("input_columns", [])
        if column in clustered.columns and pd.api.types.is_numeric_dtype(clustered[column])
    ]
    if len(numeric) >= 2:
        x_col, y_col = right.columns(2)
        x_feature = x_col.selectbox("X feature", numeric, index=0)
        y_feature = y_col.selectbox("Y feature", numeric, index=1)
        right.plotly_chart(
            px.scatter(
                clustered,
                x=x_feature,
                y=y_feature,
                color=clustered["cluster"].astype(str),
                title=f"{x_feature} vs {y_feature}",
                hover_data=[column for column in clustered.columns if column not in {x_feature, y_feature, "cluster"}][:5],
            ),
            use_container_width=True,
        )
    else:
        right.info("At least two numeric columns are needed for a feature scatter plot.")

    correlation = pipeline.phase_results.get(11, {}).get("tables", {}).get("Correlation heatmap")
    if correlation is not None and not correlation.empty:
        st.plotly_chart(
            px.imshow(correlation, text_auto=True, aspect="auto", title="Numeric feature correlations"),
            use_container_width=True,
        )


def main() -> None:
    st.title("Generic Automated Clustering Platform")
    st.caption("Upload tabular CSV data to explore, compare, and apply clustering models.")
    with st.sidebar:
        st.header("Workspace")
        selected = st.radio("Section", SIDEBAR_OPTIONS, index=0)

    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV dataset to run the 24-phase clustering workflow.")
        return

    contents = uploaded.getvalue()
    signature = hashlib.sha256(b"clustering-workflow-v2\0" + contents).hexdigest()
    try:
        df = read_csv(contents)
    except UnicodeDecodeError as exc:
        st.error(f"Could not decode the uploaded CSV. Save it as UTF-8 or Windows CSV and try again. Details: {exc}")
        return
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(f"Could not parse the uploaded CSV. Check that it is a valid comma-separated file. Details: {exc}")
        return
    if df.empty:
        st.error("The uploaded CSV has no data rows.")
        return

    if st.session_state.get("dataset_signature") != signature:
        with st.spinner("Analyzing data and fitting clustering models..."):
            pipeline = GenericClusteringPipeline()
            try:
                results = pipeline.run(df)
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"The clustering workflow could not complete: {exc}")
                st.exception(exc)
                return
        st.session_state["dataset_signature"] = signature
        st.session_state["clustering_pipeline"] = pipeline
        st.session_state["phase_results"] = results

    pipeline = st.session_state["clustering_pipeline"]
    phase_results = st.session_state["phase_results"]
    st.caption(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")

    if selected == "Dashboard":
        st.subheader("Workflow overview")
        completed = sum(result.get("status") == "completed" for result in phase_results.values())
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Input columns", len(df.columns))
        col3.metric("Phases completed", f"{completed} / 24")
        st.progress(completed / max(len(phase_results), 1))
        for phase in sorted(phase_results):
            result = phase_results[phase]
            icon = "✅" if result.get("status") == "completed" else "⚠️"
            st.write(f"{icon} **Phase {phase}: {result['name']}** — {result.get('summary', '')}")

    elif selected == "Data Understanding":
        render_phase(phase_results[2])
    elif selected == "Data Cleaning":
        render_phase(phase_results[3])
    elif selected == "Feature Engineering":
        render_phase(phase_results[6])
    elif selected == "Feature Analysis":
        render_phase(phase_results[8])
        render_phase(phase_results[11])
    elif selected == "Algorithm Comparison":
        render_phase(phase_results[14])
        render_phase(phase_results[16])
    elif selected == "Final Clusters":
        render_phase(phase_results[18])
        render_phase(phase_results[15])
    elif selected == "Cluster Profiles":
        render_phase(phase_results[19])
    elif selected == "Visualizations":
        render_phase(phase_results[20])
        render_visualizations(pipeline)
    elif selected == "Prediction":
        render_phase(phase_results[24])
        render_prediction(pipeline, df)
    elif selected == "Downloads":
        if pipeline.model_bundle.get("blob"):
            st.download_button(
                "Download trained model",
                data=pipeline.model_bundle["blob"],
                file_name="clustering-model.joblib",
                mime="application/octet-stream",
            )
            if not pipeline.clustered_data.empty:
                st.download_button(
                    "Download clustered CSV",
                    data=pipeline.clustered_data.to_csv(index=False).encode("utf-8"),
                    file_name="clustered-data.csv",
                    mime="text/csv",
                )
        else:
            st.info("No trained model bundle is available for download.")
    elif selected == "About":
        st.markdown(
            "This platform profiles tabular CSV data, preprocesses numeric and categorical "
            "features, compares clustering candidates, explains cluster profiles, and supports "
            "interactive predictions with the trained model."
        )


if __name__ == "__main__":
    main()
