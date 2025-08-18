import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor  # Changed from Classifier to Regressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
import numpy as np

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model():

    # Print this file's name in bold formatting using ASCII
    print("\n\033[1m### 4. src/train.py\033[0m ###\n")

    params = load_params()
    train_params = params['train']
    
    # Load training data
    train_data = pd.read_csv(train_params['train_data'])
    target_col = train_params['target_column']
    print(f"\tUsing target column: {target_col}")
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    print(f"\tTraining data shape: {X_train.shape}")
    print(f"\tTarget statistics:")
    print(f"\t\tMean: {y_train.mean():.4f}")
    print(f"\t\tStd: {y_train.std():.4f}")
    print(f"\t\tMin: {y_train.min():.4f}")
    print(f"\t\tMax: {y_train.max():.4f}")

    # Create regression model
    model = RandomForestRegressor(
        n_estimators=train_params['n_estimators'],
        max_depth=train_params['max_depth'],
        random_state=train_params['random_state'],
        n_jobs=-1  # Use all available cores
    )
    
    # Train model
    print("\tTraining regression model...")
    model.fit(X_train, y_train)
    
    # Cross-validation with regression scoring
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
    cv_scores_rmse = np.sqrt(-cv_scores_rmse)  # Convert to RMSE
    
    # Training performance
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, train_params['model_output'])
    
    # Save training metrics
    train_metrics = {
        "train_r2": float(train_r2),
        "train_rmse": float(train_rmse),
        "cv_r2_mean": float(cv_scores_r2.mean()),
        "cv_r2_std": float(cv_scores_r2.std()),
        "cv_rmse_mean": float(cv_scores_rmse.mean()),
        "cv_rmse_std": float(cv_scores_rmse.std()),
        "n_estimators": train_params['n_estimators'],
        "max_depth": train_params['max_depth'],
        "target_mean": float(y_train.mean()),
        "target_std": float(y_train.std())
    }
    
    os.makedirs('reports/metrics', exist_ok=True)
    with open('reports/metrics/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\tTraining R²: {train_r2:.4f}")
    print(f"\tTraining RMSE: {train_rmse:.4f}")
    print(f"\tCV R² score: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
    print(f"\tCV RMSE score: {cv_scores_rmse.mean():.4f} (+/- {cv_scores_rmse.std() * 2:.4f})")

if __name__ == "__main__":
    train_model()