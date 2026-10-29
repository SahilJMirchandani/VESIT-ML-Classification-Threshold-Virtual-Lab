# Classification Threshold Tuning — ML Virtual Lab

A focused, interactive Machine Learning virtual lab aligned to the VESIT Computer Engineering ML syllabus.

## Student details

| Team Member | Roll No. | Class | Batch | Branch | Year |
|---|---:|---|---|---|---|
| Sahil Mirchandani | 37 | D12A | A | CMPN (Computer Engineering) | T.E. (Third Year) |
| Zaid Mulla | 41 | D12A | A | CMPN (Computer Engineering) | T.E. (Third Year) |
| Malika Makhija | 31 | D12A | A | CMPN (Computer Engineering) | T.E. (Third Year) |
| Palak Lund | 30 | D12A | A | CMPN (Computer Engineering) | T.E. (Third Year) |

**Assessment:** Machine Learning Virtual Lab CA

## Live demo

- **Interactive lab:** https://vesit-ml-classification-threshold-lab.onrender.com
- **Health endpoint:** https://vesit-ml-classification-threshold-lab.onrender.com/api/health
- **Source code:** https://github.com/SahilJMirchandani/VESIT-ML-Classification-Threshold-Virtual-Lab

## Exact concept

**Classification threshold tuning in Logistic Regression.** Students move a probability cut-off and inspect how it changes predicted class labels, confusion-matrix counts, accuracy, precision, recall (sensitivity), F1-score and specificity. ROC-AUC remains fixed for a trained model because its probability rankings do not change when only the decision threshold moves.

## Dataset and methodology

- Dataset: scikit-learn's Breast Cancer Wisconsin (Diagnostic) benchmark dataset (569 samples, 30 numeric features).
- Positive class: **malignant** (explicitly remapped to target value 1 for this experiment).
- Split: stratified 60% train / 20% validation / 20% final test, `random_state=42` (341 training, 114 validation, 114 test examples).
- Model: `StandardScaler` + `LogisticRegression(max_iter=3000, random_state=42)`.
- Data leakage prevention: scaler and classifier are fitted only on the training subset.
- Threshold exploration: the slider, comparison table, and case explorer use validation predictions.
- Final evaluation: freeze the selected threshold to evaluate it once on the separate test subset; the slider/presets are locked until the experiment is restarted.
- Metrics and probabilities are computed from the actual fitted model and dataset, not hard-coded examples.

## Features

- Live validation metrics and confusion matrix.
- Preset thresholds (0.25, 0.50, 0.75) plus a continuous threshold slider.
- Comparison table showing accuracy, precision, recall, F1, false negatives and false positives.
- Validation-case explorer with filters for mistakes and predicted-positive cases.
- A deliberate final-test step that freezes the threshold and reveals test metrics and the test confusion matrix.
- CSV export for the selected validation run, plus final test results if the learner evaluated the held-out test.
- Experiment brief, step-by-step procedure, definitions, viva questions, inference and conclusion prompts.
- Health endpoint for deployment checks.

## Run locally

Python 3.10+ is recommended. Use a virtual environment to keep the dependencies isolated.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000.

Run regression tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Experiment procedure

1. Load the benchmark data and map malignant to the positive class.
2. Split data into stratified training, validation and test subsets.
3. Fit feature scaling and Logistic Regression on training data only.
4. Generate probabilities for the validation and test examples.
5. Use validation data to explore the precision/recall trade-off by moving the threshold.
6. Record the selected threshold and its validation metrics.
7. Freeze the threshold and reveal final test metrics, without using those test results to retune the threshold.

## Metric formulas

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- Specificity = TN / (TN + FP)

If a metric's denominator is zero, the lab returns 0 for that ratio.

## Endpoints

- `/` — interactive virtual lab
- `/api/data` — model metadata, split sizes and validation probabilities/labels
- `/api/evaluate-test` — POST a JSON body such as `{"threshold": 0.5}`; returns aggregate metrics for the final test set without exposing individual test cases
- `/api/health` — lightweight model/split health status

## Data provenance and authenticity

This project does **not** use fabricated/mock dataset rows, placeholder model predictions, or hard-coded classification metrics. At startup, `app.py` loads the published **Breast Cancer Wisconsin (Diagnostic)** benchmark with `sklearn.datasets.load_breast_cancer()`, fits a real `StandardScaler + LogisticRegression` pipeline on the training split, and calculates validation/test probabilities and metrics from the resulting model.

The source dataset contains 569 observations and 30 real-valued features. UCI describes these measurements as features computed from digitized fine-needle-aspirate images of breast masses; this is a historical public benchmark, **not a live hospital feed or current clinical data**. The lab assigns display labels such as `VAL-001` to validation rows for easier navigation. Those labels are generated by the lab; the underlying observation values and class labels come from the published dataset. The threshold presets (0.25, 0.50, 0.75) are experiment settings, not records or precomputed results. Metrics are calculated at runtime from the fitted model's predictions.

**Dataset attribution:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. The UCI dataset page identifies the dataset as 569 instances, 30 features, and CC BY 4.0 licensed.

- [Official UCI dataset record](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
- [scikit-learn dataset loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Safety and interpretation

This is an educational classification experiment, not a clinical decision tool. It must not be used for diagnosis or treatment. The model's scores depend on a benchmark dataset and a single deterministic split; they do not guarantee performance on future data. Threshold selection should reflect error costs and be conducted on validation data, with final evaluation performed on untouched test data.
