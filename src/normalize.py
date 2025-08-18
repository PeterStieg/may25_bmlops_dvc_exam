import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def normalize_data():
    params = load_params()
    norm_params = params['normalize']
    
    # Load train and test data
    train_data = pd.read_csv(norm_params['input_train'])
    test_data = pd.read_csv(norm_params['input_test'])
    
    # Use centralized target column
    target_col = norm_params['target_column']
    print(f"Using target column: {target_col}")
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Choose scaler
    method = norm_params['method']
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Combine with targets
    train_normalized = pd.concat([X_train_scaled, y_train], axis=1)
    test_normalized = pd.concat([X_test_scaled, y_test], axis=1)
    
    # Create output directories
    os.makedirs('data/normalized', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save normalized data
    train_normalized.to_csv(norm_params['output_train'], index=False)
    test_normalized.to_csv(norm_params['output_test'], index=False)
    
    # Save scaler
    joblib.dump(scaler, norm_params['scaler_output'])
    
    print(f"Normalization complete using {method} method")
    print(f"Train shape: {train_normalized.shape}")
    print(f"Test shape: {test_normalized.shape}")

if __name__ == "__main__":
    normalize_data()
    