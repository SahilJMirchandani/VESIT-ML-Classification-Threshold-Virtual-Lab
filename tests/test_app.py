"""Regression tests for the ML classification threshold-tuning lab."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import app  # noqa: E402


def evaluate(cases: list[dict], threshold: float) -> dict[str, float | int]:
    """Reconcile confusion-matrix counts and metrics for validation examples."""
    tp = tn = fp = fn = 0
    for case in cases:
        predicted = int(case["malignantProbability"] >= threshold)
        actual = case["actual"]
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
    }


class ModelInvariantTests(unittest.TestCase):
    def test_source_is_published_wisconsin_dataset_not_synthetic_data(self) -> None:
        # Confirm the app loads the published scikit-learn/UCI dataset and the
        # familiar class distribution instead of building mock rows.
        self.assertIn("UCI ML Breast Cancer Wisconsin (Diagnostic)", app.dataset.DESCR)
        self.assertEqual(app.X.shape, (569, 30))
        self.assertEqual(int((app.dataset.target == 0).sum()), 212)  # malignant in sklearn's dataset
        self.assertEqual(int((app.dataset.target == 1).sum()), 357)  # benign in sklearn's dataset
        self.assertEqual(
            app.LAB_DATA["dataset"]["source"],
            "UCI Machine Learning Repository; loaded via scikit-learn",
        )
        cases = app.LAB_DATA["cases"]
        self.assertEqual(len(cases), 114)
        self.assertTrue(all(c["id"] == f"VAL-{i + 1:03d}" for i, c in enumerate(cases)))
        self.assertTrue(all(c["actual"] in (0, 1) for c in cases))
        self.assertTrue(all(0.0 <= c["malignantProbability"] <= 1.0 for c in cases))

    def test_expected_dataset_and_split_sizes(self) -> None:
        meta = app.LAB_DATA["dataset"]
        self.assertEqual(meta["samples"], 569)
        self.assertEqual(meta["features"], 30)
        self.assertEqual(
            (meta["trainSamples"], meta["validationSamples"], meta["testSamples"]),
            (341, 114, 114),
        )
        self.assertEqual(
            meta["trainSamples"] + meta["validationSamples"] + meta["testSamples"],
            meta["samples"],
        )

    def test_stratified_splits_keep_all_examples(self) -> None:
        meta = app.LAB_DATA["dataset"]
        self.assertEqual(meta["validationMalignant"] + meta["validationBenign"], meta["validationSamples"])
        self.assertEqual(int(app.y_test.sum()) + int(len(app.y_test) - app.y_test.sum()), meta["testSamples"])
        for key in ("validationMalignant", "validationBenign"):
            self.assertGreater(meta[key], 0)
        self.assertGreater(int(app.y_test.sum()), 0)
        self.assertGreater(int(len(app.y_test) - app.y_test.sum()), 0)

    def test_validation_probabilities_and_auc_are_valid(self) -> None:
        cases = app.LAB_DATA["cases"]
        self.assertEqual(len(cases), 114)
        for case in cases:
            self.assertIn(case["actual"], (0, 1))
            self.assertGreaterEqual(case["malignantProbability"], 0.0)
            self.assertLessEqual(case["malignantProbability"], 1.0)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in app.test_scores))
        self.assertGreaterEqual(app.LAB_DATA["model"]["validationRocAuc"], 0.0)
        self.assertLessEqual(app.LAB_DATA["model"]["validationRocAuc"], 1.0)
        self.assertGreaterEqual(app.test_auc, 0.0)
        self.assertLessEqual(app.test_auc, 1.0)

    def test_threshold_counts_reconcile_and_positive_predictions_are_monotonic(self) -> None:
        cases = app.LAB_DATA["cases"]
        low, middle, high = (evaluate(cases, t) for t in (0.25, 0.50, 0.75))
        for result in (low, middle, high):
            self.assertEqual(result["tp"] + result["fn"], app.LAB_DATA["dataset"]["validationMalignant"])
            self.assertEqual(result["tn"] + result["fp"], app.LAB_DATA["dataset"]["validationBenign"])
            self.assertEqual(result["tp"] + result["tn"] + result["fp"] + result["fn"], 114)
            for key in ("accuracy", "precision", "recall", "specificity", "f1"):
                self.assertGreaterEqual(result[key], 0.0)
                self.assertLessEqual(result[key], 1.0)
        self.assertGreaterEqual(low["tp"] + low["fp"], middle["tp"] + middle["fp"])
        self.assertGreaterEqual(middle["tp"] + middle["fp"], high["tp"] + high["fp"])

    def test_final_test_predictions_are_not_in_browser_payload(self) -> None:
        self.assertIn("cases", app.LAB_DATA)
        self.assertNotIn("testCases", app.LAB_DATA)
        self.assertNotIn("testMalignant", app.LAB_DATA["dataset"])
        self.assertNotIn("testBenign", app.LAB_DATA["dataset"])
        self.assertNotIn("testRocAuc", app.LAB_DATA["model"])
        val_ids = {row["id"] for row in app.LAB_DATA["cases"]}
        self.assertTrue(all(key.startswith("VAL-") for key in val_ids))


class HTTPEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.LabHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_endpoint_reports_split(self) -> None:
        with urlopen(self.base_url + "/api/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "trained")
        self.assertEqual(payload["trainSamples"], 341)
        self.assertEqual(payload["validationSamples"], 114)
        self.assertEqual(payload["testSamples"], 114)

    def test_data_endpoint_only_exposes_validation_rows(self) -> None:
        with urlopen(self.base_url + "/api/data", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(len(payload["cases"]), 114)
        self.assertNotIn("testCases", payload)
        self.assertIn("validationRocAuc", payload["model"])
        self.assertNotIn("testRocAuc", payload["model"])
        self.assertNotIn("testMalignant", payload["dataset"])
        self.assertNotIn("testBenign", payload["dataset"])

    def test_final_test_endpoint_returns_aggregate_metrics_only(self) -> None:
        request = Request(
            self.base_url + "/api/evaluate-test",
            data=json.dumps({"threshold": 0.50}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(payload["total"], 114)
        self.assertEqual(payload["tp"] + payload["tn"] + payload["fp"] + payload["fn"], 114)
        self.assertIn("rocAuc", payload)
        self.assertNotIn("cases", payload)
        self.assertNotIn("probabilities", payload)

    def test_invalid_threshold_is_rejected(self) -> None:
        request = Request(
            self.base_url + "/api/evaluate-test",
            data=json.dumps({"threshold": 2.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 400)

    def test_root_serves_the_interactive_lab(self) -> None:
        with urlopen(self.base_url + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        self.assertIn("Classification", html)
        self.assertIn("threshold-slider", html)
        self.assertIn("test-eval-btn", html)
        self.assertIn("export-btn", html)

    def test_unknown_routes_return_404(self) -> None:
        with self.assertRaises(HTTPError) as context:
            urlopen(self.base_url + "/not-a-real-route", timeout=3)
        self.assertEqual(context.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
