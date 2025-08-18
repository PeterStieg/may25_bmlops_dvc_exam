import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import numpy as np

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def identify_column_types(df, target_col):
    """Identify numeric, datetime, and categorical columns for proper processing."""
    
    # Exclude target column from feature analysis
    feature_cols = [col for col in df.columns if col != target_col]
    
    numeric_cols = []
    datetime_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        # Check if column contains datetime-like strings
        elif df[col].dtype == 'object':
            # Try to parse a sample of values to detect datetime
            sample_values = df[col].dropna().head(10)
            datetime_detected = False
            
            for val in sample_values:
                try:
                    pd.to_datetime(val)
                    datetime_detected = True
                    break
                except (ValueError, TypeError):
                    continue
            
            if datetime_detected:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, datetime_cols, categorical_cols

def normalize_data():

    # Print this file's name in bold formatting using ASCII
    print("\n\033[1m### 3. src/normalize.py ###\033[0m\n")

    params = load_params()
    norm_params = params['normalize']
    
    # Load train and test data
    train_data = pd.read_csv(norm_params['input_train'])
    test_data = pd.read_csv(norm_params['input_test'])
    
    # Use centralized target column
    target_col = norm_params['target_column']
    print(f"\tUsing target column: {target_col}")
    
    print(f"\tOriginal train data shape: {train_data.shape}")
    print(f"\tOriginal test data shape: {test_data.shape}")
    
    # Analyze column types to handle mixed data properly
    numeric_cols, datetime_cols, categorical_cols = identify_column_types(train_data, target_col)
    
    print(f"\tColumn Analysis:")
    print(f"\t\tNumeric columns ({len(numeric_cols)}): {numeric_cols[:5]}...")  # Show first 5
    print(f"\t\tDatetime columns ({len(datetime_cols)}): {datetime_cols}")
    print(f"\t\tCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"\t\tTarget column: {target_col}")
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Extract only numeric features for normalization
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for normalization!")
    
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    print(f"\tNumeric features for normalization:")
    print(f"\t\tShape: {X_train_numeric.shape}")
    print(f"\t\tColumns: {list(X_train_numeric.columns)}")

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
    
    print(f"\tApplying {method} normalization...")
    
    # Fit and transform only numeric columns
    X_train_scaled_numeric = pd.DataFrame(
        scaler.fit_transform(X_train_numeric),
        columns=X_train_numeric.columns,
        index=X_train_numeric.index
    )
    
    X_test_scaled_numeric = pd.DataFrame(
        scaler.transform(X_test_numeric),
        columns=X_test_numeric.columns,
        index=X_test_numeric.index
    )
    
    # Handle non-numeric columns (excluded for now)
    if len(datetime_cols) > 0:
        print(f"\tExcluding {len(datetime_cols)} datetime columns from output")
    if len(categorical_cols) > 0:
        print(f"\tExcluding {len(categorical_cols)} categorical columns from output")
    
    print(f"\t\tNote: Only numeric columns are included in normalized output.")
    print(f"\t\tConsider feature engineering for datetime/categorical columns in future iterations.")
    
    # Combine scaled numeric features with targets
    train_normalized = pd.concat([X_train_scaled_numeric, y_train], axis=1)
    test_normalized = pd.concat([X_test_scaled_numeric, y_test], axis=1)
    
    # Create output directories (defensive programming)
    os.makedirs('data/normalized', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save normalized data
    train_normalized.to_csv(norm_params['output_train'], index=False)
    test_normalized.to_csv(norm_params['output_test'], index=False)
    
    # Save scaler for inference pipeline
    joblib.dump(scaler, norm_params['scaler_output'])
    
    # Save column information for later pipeline stages
    column_info = {
        'numeric_columns': numeric_cols,
        'datetime_columns': datetime_cols,
        'categorical_columns': categorical_cols,
        'target_column': target_col,
        'normalization_method': method
    }
    
    with open('models/column_info.json', 'w') as f:
        import json
        json.dump(column_info, f, indent=2)
    
    print(f"\tNormalization complete!")
    print(f"\t\tMethod: {method}")
    print(f"\t\tTrain shape: {train_normalized.shape}")
    print(f"\t\tTest shape: {test_normalized.shape}")
    print(f"\t\tFeatures normalized: {len(numeric_cols)}")

if __name__ == "__main__":
    normalize_data()
    