from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from backend.pipeline import GenericClusteringPipeline
from backend.utils.csv_reader import read_csv_bytes
from backend.utils.preprocessing import predict_cluster_labels


class GenericClusteringPipelineTests(unittest.TestCase):
    def test_csv_reader_accepts_windows_1252_encoding(self) -> None:
        contents = "name,city\nAndré,„Paris”\n".encode("cp1252")

        frame = read_csv_bytes(contents)

        self.assertEqual(frame.loc[0, "name"], "André")
        self.assertEqual(frame.loc[0, "city"], "„Paris”")

    def test_csv_reader_accepts_utf8_bom(self) -> None:
        contents = "\ufeffname,value\nsample,12\n".encode("utf-8")

        frame = read_csv_bytes(contents)

        self.assertEqual(frame.columns.tolist(), ["name", "value"])
        self.assertEqual(frame.loc[0, "value"], 12)

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "account_id": range(100, 124),
                "monthly_spend": [10, 12, 11, 13, 9, 14, 80, 84, 82, 79, 88, 81] * 2,
                "visits": [1, 2, 1, 3, 2, 1, 8, 9, 8, 7, 9, 8] * 2,
                "segment": ["basic"] * 6 + ["premium"] * 6 + ["basic"] * 6 + ["premium"] * 6,
            }
        )
        self.data.loc[0, "monthly_spend"] = np.nan
        self.data.loc[1, "segment"] = None

    def test_runs_all_phases_and_handles_mixed_features(self) -> None:
        pipeline = GenericClusteringPipeline()
        results = pipeline.run(self.data)

        self.assertEqual(sorted(results), list(range(1, 25)))
        self.assertEqual(results[10]["status"], "completed")
        self.assertEqual(results[12]["status"], "completed")
        self.assertEqual(results[18]["status"], "completed")
        self.assertEqual(results[19]["status"], "completed")
        self.assertIn("blob", pipeline.model_bundle)
        self.assertIn("segment", pipeline.model_bundle["categorical_features"])
        self.assertEqual(len(pipeline.clustered_data), len(self.data))
        self.assertEqual(
            {row["algorithm"] for row in results[14]["tables"]["Metrics"].to_dict("records")},
            {
                "KMeans",
                "GaussianMixture",
                "MeanShift",
                "Hierarchical Clustering",
                "DBSCAN",
            },
        )

    def test_prediction_supports_unseen_category_and_missing_optional_columns(self) -> None:
        pipeline = GenericClusteringPipeline()
        pipeline.run(self.data)

        prediction_input = pd.DataFrame(
            {"monthly_spend": [30.0], "visits": [4], "segment": ["new-segment"]}
        )
        predictions = pipeline.predict(prediction_input)

        self.assertEqual(predictions.shape, (1,))
        self.assertIn(int(predictions[0]), pipeline.clustered_data["cluster"].unique())

    def test_repeated_run_replaces_previous_artifacts(self) -> None:
        pipeline = GenericClusteringPipeline()
        pipeline.run(self.data)
        first_blob = pipeline.model_bundle["blob"]
        pipeline.run(self.data.assign(segment="same"))

        self.assertEqual(sorted(pipeline.phase_results), list(range(1, 25)))
        self.assertTrue(pipeline.model_bundle["blob"])
        self.assertIsInstance(first_blob, bytes)

    def test_categorical_only_dataset_can_be_clustered_and_predicted(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["north", "south", "east", "west"] * 6,
                "plan": ["free", "free", "pro", "pro"] * 6,
            }
        )
        pipeline = GenericClusteringPipeline()
        results = pipeline.run(data)

        self.assertEqual(results[18]["status"], "completed")
        self.assertEqual(pipeline.predict(pd.DataFrame({"region": ["central"], "plan": ["free"]})).shape, (1,))

    def test_non_predictive_algorithms_support_new_row_assignment(self) -> None:
        reference = pd.DataFrame({"x": [0.0, 0.1, 5.0, 5.1], "y": [0.0, 0.2, 5.0, 4.9]})
        new_rows = pd.DataFrame({"x": [0.05, 5.05], "y": [0.1, 5.0]})
        for model in (
            AgglomerativeClustering(n_clusters=2),
            DBSCAN(eps=0.5, min_samples=2),
        ):
            labels = model.fit_predict(reference)
            predicted = predict_cluster_labels(model, new_rows, reference, labels)
            self.assertEqual(predicted.shape, (2,))


if __name__ == "__main__":
    unittest.main()
