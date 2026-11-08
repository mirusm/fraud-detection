from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def evaluate(model, X_part, y_part):
    probabilities = model.predict_proba(X_part)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    return {
        'PR-AUC (Average Precision)': average_precision_score(y_part, probabilities),
        'ROC-AUC': roc_auc_score(y_part, probabilities),
        'Recall': recall_score(y_part, predicted, zero_division=0),
        'Precision': precision_score(y_part, predicted, zero_division=0),
        'F1': f1_score(y_part, predicted, zero_division=0),
    }, confusion_matrix(y_part, predicted, labels=[0, 1])
