import time
import numpy as np
import optuna
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

def create_model_pipeline(preprocessor, model=None) -> Pipeline:
    """
    Returns a pipeline with preprocessing and a model.
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def optimize_hyperparameters(X, y, preprocessor, n_trials: int = 30) -> dict:
    """
    Uses Optuna to find the best RandomForest hyperparameters.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }

        model = create_model_pipeline(preprocessor, RandomForestClassifier(
            **params, class_weight='balanced', random_state=42))

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def select_features(X_processed, y, model: BaseEstimator, feature_names: list[str]) -> tuple[list[str], RFECV]:
    """
    Applies RFECV to select the most important features.
    """
    selector = RFECV(
        estimator=model,
        step=1,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    selector.fit(X_processed, y)

    selected = np.array(feature_names)[selector.support_]
    return selected.tolist(), selector

def explain_with_shap(model: BaseEstimator, X_selected, feature_names: list[str]):
    """
    Uses SHAP to visualize feature importance.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)
    shap.summary_plot(shap_values[1], X_selected, feature_names=feature_names, show=True)

def final_modeling_pipeline(X_processed, y, feature_names, selected_features, eda_features, best_params):
    # Combine features
    combined = list(set(selected_features + eda_features))
    combined_idx = [i for i, f in enumerate(feature_names) if f in combined]

    X_final = X_processed[:, combined_idx]
    final_features = [feature_names[i] for i in combined_idx]

    # Train with best hyperparameters
    model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    model.fit(X_final, y)

    return model, X_final, final_features

def train_final_model(X, y, best_params) -> BaseEstimator:
    model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model

def map_eda_to_ohe(eda_features, all_feature_names):
    mapped = []
    for eda in eda_features:
        matches = [f for f in all_feature_names if f.endswith(eda) or eda in f]
        mapped.extend(matches)
    return list(set(mapped))

def evaluate_features_by_count(model, X, y, feature_names, max_features=30, step=5, verbose=True):
    scores = []
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    # Extract model hyperparameters once (not reusing the trained model object)
    model_params = {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'class_weight': model.class_weight,
        'random_state': 42,
        'n_jobs': -1
    }

    for k in range(step, max_features + 1, step):
        top_k_idx = sorted_idx[:k]
        X_k = X[:, top_k_idx]

        clf = RandomForestClassifier(**model_params)

        time.time()
        score = cross_val_score(clf, X_k, y, cv=2, scoring='roc_auc').mean()
        time.time()

        scores.append((k, score))

        if verbose:
            pass

    return scores, sorted_idx
