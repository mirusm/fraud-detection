import numpy as np
from sklearn.metrics import average_precision_score

from .train import make_model


def suggest_xgboost_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 300.0, log=True),
    }


def make_tuned_xgboost(columns, params, *, categorical_columns):
    model = make_model('XGBoost', columns, categorical_columns=categorical_columns)
    model.set_params(**{f'classifier__{key}': value for key, value in params.items()})

    return model


def optuna_objective(trial, *, X, y, tuning_folds, categorical_columns):
    params = suggest_xgboost_params(trial)
    scores = []

    for fold_train_index, fold_validation_index, fold_features in tuning_folds:
        model = make_tuned_xgboost(fold_features, params, categorical_columns=categorical_columns)
        model.fit(X.loc[fold_train_index, fold_features], y.loc[fold_train_index])
        probabilities = model.predict_proba(X.loc[fold_validation_index, fold_features])[:, 1]
        scores.append(average_precision_score(y.loc[fold_validation_index], probabilities))
    trial.set_user_attr('fold_average_precision', scores)

    return float(np.mean(scores))


def select_precision_threshold(y_true, probabilities, min_precision):
    y_array = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_array) != len(probabilities) or len(y_array) == 0:
        raise ValueError('Labels and probabilities must have the same nonzero length.')
    if not 0 < min_precision <= 1:
        raise ValueError('Minimum precision must be greater than zero and at most one.')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Predicted probabilities must be finite and between zero and one.')
    if not np.isin(y_array, [0, 1]).all() or y_array.sum() == 0:
        raise ValueError('Threshold selection requires binary labels with at least one fraud.')
    
    order = np.argsort(-probabilities, kind='stable')
    sorted_probabilities = probabilities[order]
    sorted_labels = y_array[order]
    group_ends = np.flatnonzero(np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True])
    counts = group_ends + 1
    true_positives = np.cumsum(sorted_labels)[group_ends]
    valid = true_positives / counts >= min_precision

    if not valid.any():
        raise ValueError('No non-empty alert set satisfies the minimum precision; review the requirement or tied scores.')
    
    valid_positions = np.flatnonzero(valid)
    best_position = valid_positions[np.argmax(true_positives[valid_positions])]

    return float(sorted_probabilities[group_ends[best_position]])
