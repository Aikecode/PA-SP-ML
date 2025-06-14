from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_base_model(model_name: str):

    if model_name == 'LogisticRegression':
        return LogisticRegression(random_state=42)
    elif model_name == 'SVC':
        return SVC(random_state=42, probability=True)
    elif model_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'AdaBoost':
        return AdaBoostClassifier(random_state=42)
    elif model_name == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=42)
    elif model_name == 'XGBoost':
        return XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss')
    elif model_name == 'LGBM':
        return LGBMClassifier(random_state=42, objective='binary')
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                         "Available models are: 'LogisticRegression', 'SVC', 'DecisionTree', "
                         "'RandomForest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LGBM'.")

def create_model_with_params(model_name: str, params: dict):

    base_model = get_base_model(model_name)
    try:
        base_model.set_params(**params)
    except TypeError as e:
        raise TypeError(f"Error setting parameters for {model_name}: {e}. "
                        f"Please check if the provided parameters are valid for this model: {params}")
    return base_model