import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold 

from src.data_processing import load_and_split_data
from src.models import create_model_with_params, get_base_model

X_train, X_test, y_train, y_test, feature_names = load_and_split_data(
    outcome_type='pa',
    data_purpose='modeling'
)

base_estimator_ada_params = {
    'criterion': 'entropy',
    'max_depth': 4,
    'max_features': None,
    'min_samples_leaf': 5,
    'min_samples_split': 5,
    'random_state': 42
}
base_estimator_ada_instance = create_model_with_params('DecisionTree', base_estimator_ada_params)

models_to_compare = {
    'LogisticRegression': {
        'params': {'penalty': "l2", 'C': 0.05, 'solver': "lbfgs", 'random_state': 42},
        'model_name_for_function': 'LogisticRegression'
    },
    'DecisionTree': {
        'params': {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 5, 'random_state': 42},
        'model_name_for_function': 'DecisionTree' 
    },
    'SVC': {
        'params': {'kernel': 'rbf', 'C': 0.01, 'gamma': 'scale', 'probability': True, 'random_state': 42},
        'model_name_for_function': 'SVC'
    },
    'RandomForest': {
        'params': {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 275, 'random_state': 42},
        'model_name_for_function': 'RandomForest'
    },
    'AdaBoost': {
        'params': {'n_estimators': 16, 'learning_rate': 0.03, 'estimator': base_estimator_ada_instance, 'algorithm':'SAMME.R', 'random_state': 42},
        'model_name_for_function': 'AdaBoost'
    },
    'GradientBoosting': {
        'params': {'loss': 'log_loss', 'learning_rate': 0.01, 'n_estimators': 221, 'subsample': 0.6, 'criterion': 'friedman_mse', 'min_samples_split': 2, 'min_samples_leaf': 6, 'max_depth': 2, 'random_state': 42},
        'model_name_for_function': 'GradientBoosting'
    },
    'XGBoost': {
        'params': {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 206, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'gamma': 0.7, 'subsample': 1.0, 'colsample_bytree': 1.0, 'random_state': 42},
        'model_name_for_function': 'XGBoost'
    },
    'LightGBM': {
        'params': {'boosting_type': 'gbdt', 'colsample_bytree': 1.0, 'objective': 'binary', 'num_leaves': 2, 'max_depth': 6, 'learning_rate': 0.25, 'n_estimators': 50, 'subsample': 0.6, 'random_state': 42},
        'model_name_for_function': 'LGBM' 
    }
}

results = []

for model_name, model_info in models_to_compare.items():
    print(f"\n--- Evaluating {model_name} ---")

    model = create_model_with_params(model_info['model_name_for_function'], model_info['params'])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"): 
        y_prob = model.decision_function(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan 

    print(f"  Test set evaluation metrics for {model_name}:")
    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    F1 Score: {test_f1:.4f}")
    print(f"    AUC:      {test_auc:.4f}")