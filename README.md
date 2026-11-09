# Credit Card Fraud Detection

## Project Overview

This proof-of-concept project explores machine learning for identifying
potentially fraudulent card transactions. It combines exploratory analysis,
behavioral feature engineering and model evaluation on chronologically split
data. The goal is to detect fraud while considering the trade-off between
recall and false alerts.

## Dataset

The analysis expects a CSV at `data/raw/case_cc_fraud.csv`. The dataset is not
included in the repository. The target column is `is_fraud`, and fraud transactions
represent a small minority of the data. The input is sorted by timestamp for
the analysis rather than assumed to be in chronological order.

## Repository Structure

- [`data/raw/`](data/raw/) – location for the input CSV dataset.
- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) – data audit and EDA.
- [`notebooks/02_feature_exploration.ipynb`](notebooks/02_feature_exploration.ipynb)
  – feature engineering, training-only quality filters and mutual information.
- [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) – model
  comparison, tuning, threshold selection, evaluation and SHAP.
- [`src/`](src/) – shared preprocessing, training, evaluation and optimization
  functions used by the notebooks.
- [`ppt/`](ppt/) – project presentation in PPTX and PDF formats.
- [`requirements.txt`](requirements.txt) – Python dependencies.

## Methodology

1. Audit the data and explore fraud patterns.
2. Create temporal, transaction-amount, behavioral, geographic and merchant
   features. Historical features use strictly earlier transactions; events
   with the same timestamp do not enter each other's history.
3. Split chronologically into approximately **70% train, 15% validation and
   15% test**, keeping identical timestamps together.
4. Compare Logistic Regression, Random Forest and XGBoost using each model's
   training-derived feature ranking and top 10, 15, 20 and all features.
   Average precision on validation is the primary selection metric.
5. Tune XGBoost with its train-derived top 20 features using Optuna and three
   expanding time folds inside train. Each fold derives its feature ranking
   from its own training portion. The search includes `scale_pos_weight`.
6. Select a threshold on validation that maximizes recall subject to at least
   **85% precision**. Refit the tuned model on train + validation and evaluate
   on test.

Precision, recall, F1, ROC-AUC, average precision and confusion matrices are
reported. SHAP provides global explanations of the final model, including a
separate analysis of actual test frauds.

## Results

**XGBoost with 20 features** achieved the best result in the validation
comparison. After hyperparameter tuning and threshold optimization, the final
model achieved the following results on the test set:

| Metric | Value |
|---|---:|
| Average precision | 0.9786 |
| Recall | 96.75% |
| Precision | 82.78% |
| F1 | 0.8922 |
| Threshold | 0.039263 |
| True positives / false negatives | 894 / 30 |
| False positives / true negatives | 186 / 276 750 |

The 85% precision requirement applied to validation. It is not guaranteed after
refitting or on test.

## Limitations

- Fraud confirmation times and the availability of all input fields at
  prediction time are unknown. Historical features assume earlier transaction
  events are available, without using their fraud labels.
- Earlier EDA used the available dataset, and the analysis is offline. Results
  do not establish production performance, prevented losses or real-world
  generalization.
- Further work could validate on a new unseen future period and investigate
  calibration, monitoring, operational capacity and real-world data.

## Installation and Usage

Clone the repository and install the dependencies:

```powershell
git clone https://github.com/mirusm/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

Place the dataset at `data/raw/case_cc_fraud.csv`.

Open the notebooks in the order listed above and run their cells. The full
dataset and intermediate results must fit in memory, model training and the
15-trial Optuna search can take substantial time. The notebooks use a fixed
random seed where applicable, but results can vary across environments.
