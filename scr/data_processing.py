import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATHS = {
    'pa': 'data/', 
    'sp': 'data/',  
}

def load_and_split_data(
    outcome_type: str,  # 'pa' for Physical Activity, 'sp' for Sports Participation
    data_purpose: str = 'modeling', # 'modeling' for X,y split; 'pfi' for PFI; 'pdp' for PDP
    test_size: float = 0.2,
    random_state: int = 42
):
    
    if outcome_type == 'pa':
        target_column = 'physical_activity'
        data_path = DATA_PATHS['pa'] 
    elif outcome_type == 'sp':
        target_column = 'sports_participation'
        data_path = DATA_PATHS['sp'] 
    else:
        raise ValueError(f"Invalid outcome_type: {outcome_type}. Must be 'pa' or 'sp'.")

    print(f"Loading data from: {data_path} for {outcome_type} - {data_purpose}")
    data = pd.read_stata(data_path)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data from {data_path}")

    y = data[target_column]
    X = data.drop(target_column, axis=1)
    feature_names = X.columns.tolist() 

    if data_purpose in ['modeling', 'pfi']:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, feature_names
    elif data_purpose == 'pdp':
        return X, y, feature_names
    else:
        raise ValueError(f"Unknown data_purpose: {data_purpose}")
