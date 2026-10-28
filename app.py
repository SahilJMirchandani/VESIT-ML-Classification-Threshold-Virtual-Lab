"""VESIT ML Virtual Lab: Classification Threshold Tuning.

Trains Logistic Regression on scikit-learn's Breast Cancer Wisconsin dataset.
Thresholds are explored on validation predictions. Final test evaluation runs
server-side and returns metrics only, not individual test cases or probabilities.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
RANDOM_STATE = 42

# scikit-learn's source dataset uses target 0 = malignant and target 1 = benign.
# Re-map malignant to positive class 1 for the threshold-tuning experiment.
dataset = load_breast_cancer()
X = dataset.data
y_malignant = (dataset.target == 0).astype(int)

# Stratified 60/20/20 train/validation/final-test split.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y_malignant,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y_malignant,
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_remaining,
)

# The scaler and classifier are fitted only on training rows.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
)
model.fit(X_train, y_train)
malignant_class_index = list(model.classes_).index(1)
validation_scores = model.predict_proba(X_validation)[:, malignant_class_index]
test_scores = model.predict_proba(X_test)[:, malignant_class_index]
validation_auc = float(roc_auc_score(y_validation, validation_scores))
test_auc = float(roc_auc_score(y_test, test_scores))


def cases_for(y_true: np.ndarray, scores: np.ndarray, prefix: str) -> list[dict]:
    """Return validation probabilities and ground truth for the browser lab."""
    return [
        {
            "id": f"{prefix}-{i + 1:03d}",
            "actual": int(y_true[i]),
            "malignantProbability": float(scores[i]),
        }
        for i in range(len(y_true))
    ]


def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float, auc: float) -> dict:
    """Compute only aggregate metrics for a selected threshold."""
    predicted = scores >= threshold
    actual_positive = y_true == 1
    actual_negative = y_true == 0
    tp = int(np.sum(actual_positive & predicted))
    fn = int(np.sum(actual_positive & ~predicted))
    fp = int(np.sum(actual_negative & predicted))
    tn = int(np.sum(actual_negative & ~predicted))
    total = tp + tn + fp + fn

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "rocAuc": auc,
    }


# The browser receives validation cases for threshold exploration only.
# Test scores and labels are not included in this payload.
LAB_DATA = {
    "dataset": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "trainSamples": int(X_train.shape[0]),
        "validationSamples": int(X_validation.shape[0]),
        "testSamples": int(X_test.shape[0]),
        "validationMalignant": int(y_validation.sum()),
        "validationBenign": int(len(y_validation) - y_validation.sum()),
        "split": "60% train / 20% validation / 20% test",
        "randomState": RANDOM_STATE,
        "source": "scikit-learn built-in benchmark dataset",
    },
    "model": {
        "name": "Logistic Regression",
        "preprocessing": "StandardScaler fitted on training data only",
        "positiveClass": "Malignant",
        "thresholdDefault": 0.50,
        "validationRocAuc": validation_auc,
    },
    "cases": cases_for(y_validation, validation_scores, "VAL"),
}


class LabHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            payload = json.dumps({
                "status": "ok",
                "model": "trained",
                "trainSamples": len(X_train),
                "validationSamples": len(X_validation),
                "testSamples": len(X_test),
            }).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        if path == "/api/data":
            body = json.dumps(LAB_DATA, separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            try:
                page = (ROOT / "index.html").read_bytes()
            except OSError:
                self._send(500, b"Lab interface is missing.", "text/plain; charset=utf-8")
                return
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/evaluate-test":
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            request_body = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_threshold = request_body.get("threshold") if isinstance(request_body, dict) else None
            if isinstance(raw_threshold, bool):
                raise ValueError("Threshold must be numeric")
            threshold = float(raw_threshold)
            if not np.isfinite(threshold) or not 0.05 <= threshold <= 0.95:
                raise ValueError("Threshold must be between 0.05 and 0.95")
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send(400, b'{"error":"Provide a numeric threshold between 0.05 and 0.95"}', "application/json; charset=utf-8")
            return

        # Only aggregate metrics cross the API boundary for the final test set.
        result = evaluate_threshold(y_test, test_scores, threshold, test_auc)
        body = json.dumps(result, separators=(",", ":")).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        # Do not log request bodies, probabilities, or individual classifications.
        print(f"[ml-lab] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), LabHandler)
    print(
        f"ML Threshold Tuning Lab listening on port {PORT} | "
        f"{len(X_train)} train / {len(X_validation)} validation / {len(X_test)} test | "
        f"validation ROC-AUC={validation_auc:.4f} | final test ROC-AUC={test_auc:.4f}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
