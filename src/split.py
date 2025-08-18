# src/split.py
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def split_data():
    params = load_params()
    split_params = params['data_split']
    
    # Load processed data
    data = pd.read_csv(split_params['input'])
    print(f"Loaded data shape: {data.shape}")
    
    # Identify target column (use the last column if 'target' doesn't exist)
    target_col = split_params['target_column']
    if target_col not in data.columns:
        target_col = data.columns[-1]  # Use last column as target
        print(f"Target column '{split_params['target_column']}' not found. Using '{target_col}'")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_params['test_size'],
        random_state=split_params['random_state'],
        stratify=y if len(y.unique()) <= 10 else None
    )
    
    # Combine features and target
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Create output directory
    os.makedirs('data/splits', exist_ok=True)
    
    # Save splits
    train_data.to_csv('data/splits/train.csv', index=False)
    test_data.to_csv('data/splits/test.csv', index=False)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")

if __name__ == "__main__":
    split_data()
