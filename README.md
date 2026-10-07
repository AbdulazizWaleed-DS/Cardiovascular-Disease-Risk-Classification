# Cardiovascular Disease Risk Classification

**Type:** Academic project (University of Technology Sydney) — supervised classification coursework.

Binary classification of cardiovascular disease presence (`cardio`: 0 = No, 1 = Yes) from physiological, clinical, and lifestyle attributes, using the public [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (70,000 patient records). The full pipeline — data cleaning, exploratory analysis, model comparison, hyperparameter tuning, and evaluation — is implemented in `Cardiovascular_Disease_Classification.ipynb`.

## Data

- **Source:** Cardiovascular Disease dataset (Kaggle), semicolon-separated, 70,000 records, 12 features + target.
- **Features:** age (days), gender, height, weight, systolic/diastolic blood pressure (`ap_hi`/`ap_lo`), cholesterol, glucose, smoking, alcohol use, physical activity.
- **Cleaning:** removed 1,289 records with physiologically implausible blood-pressure readings (non-positive values, systolic < diastolic, systolic > 300 or diastolic > 200 mmHg), leaving 68,711 records.
- **Engineered features:** age converted from days to years; BMI computed from height and weight.

## Methodology

1. Preprocessing: outlier removal, BMI engineering, feature scaling (StandardScaler) for scale-sensitive models.
2. Exploratory analysis: distribution checks, correlation heatmap, log-transform and binning experiments on secondary features (see `notebook` — these exploratory encodings were **not** used in the final model; a simple logistic-regression check showed they added no predictive value, ~0.58 accuracy using height/weight alone).
3. Model comparison: four classifiers trained on an 80/20 stratified train/validation split — Logistic Regression, Random Forest, SVM, XGBoost.
4. Hyperparameter tuning: `GridSearchCV` (5-fold stratified) over XGBoost's `max_depth`, `learning_rate`, `n_estimators`.

## Results

Validation-set performance (80/20 stratified split, 68,711 records):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.759 | 0.665 | 0.709 |
| Random Forest | 0.733 | 0.763 | 0.667 | 0.712 |
| SVM | 0.734 | 0.766 | 0.667 | 0.713 |
| **XGBoost (best)** | **0.735** | 0.755 | **0.686** | **0.719** |

**Best model:** XGBoost, tuned via grid search (`max_depth=4, learning_rate=0.2, n_estimators=100`), achieving **AUC = 0.803** on the validation set.

**Confusion matrix (tuned XGBoost, validation set, n=13,743):**

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 5,446 | 1,498 |
| **Actual: Yes** | 2,160 | 4,639 |

**Feature importance (XGBoost):** systolic blood pressure (`ap_hi`) dominates by a wide margin, followed by cholesterol and age; physical activity, smoking, and glucose contribute modestly. This ordering is consistent with established cardiovascular risk factors in the clinical literature.

## Limitations

- Recall on the positive (disease) class (0.686) is lower than precision — in a screening context, the clinically relevant failure mode is false negatives, so this is the metric most worth improving further (e.g., via class-weighting or threshold tuning) before any real-world use.
- No external validation dataset was used; performance is reported on a held-out split from the same source distribution only.
- This run does not include a separate unlabeled test-set submission step — the pipeline supports one (see `notebook`), but it was not exercised here.

## Repository Contents

- `Cardiovascular_Disease_Classification.ipynb` — full pipeline: data loading, cleaning, EDA, model comparison, tuning, evaluation.
- `requirements.txt` — Python dependencies.

## Running It

```bash
pip install -r requirements.txt
```

Place the dataset as `cardio_train.csv` (semicolon-separated) in the repository root, then run the notebook top to bottom.

## License

MIT
