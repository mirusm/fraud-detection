import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


def make_model(model_name, columns, *, categorical_columns):
    transformers = []

    for i, column in enumerate(columns):
        if column in categorical_columns:
            encoder = (OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
                       if model_name == 'Logistic Regression'
                       else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32))
            steps = [('imputer', SimpleImputer(strategy='constant', fill_value='__missing__', keep_empty_features=True)),
                     ('encoder', encoder)]
        else:
            steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]

        transformers.append((f'feature_{i}', Pipeline(steps), [column]))

    preprocessor = ColumnTransformer(transformers, sparse_threshold=1.0, verbose_feature_names_out=False)

    if model_name == 'Logistic Regression':
        return Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
        ])
    
    if model_name == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        classifier = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    return Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])


def feature_importance(model, columns):
    classifier = model.named_steps['classifier']

    if hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
    else:
        values = classifier.feature_importances_

    indices = model.named_steps['preprocessor'].output_indices_
    scores = [values[indices[f'feature_{i}']].sum() for i in range(len(columns))]

    return pd.DataFrame({'feature': columns, 'importance': scores}).sort_values(
        'importance', ascending=False, kind='stable'
    ).reset_index(drop=True)
