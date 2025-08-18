# src/train.py
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import json
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model():
    params = load_params()
    train_params = params['train']
    
    # Load training data
    train_data = pd.read_csv(train_params['train_data'])
    target_col = train_data.columns[-1]  # Assume last column is target
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=train_params['n_estimators'],
        max_depth=train_params['max_depth'],
        random_state=train_params['random_state']
    )
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    train_accuracy = model.score(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, train_params['model_output'])
    
    # Save training metrics
    train_metrics = {
        "train_accuracy": float(train_accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std())
    }
    
    os.makedirs('reports/metrics', exist_ok=True)
    with open('reports/metrics/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"Training complete!")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if __name__ == "__main__":
    train_model()
    