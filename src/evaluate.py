import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_params():
    """Load parameters from params.yaml file."""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model():
    
    print("\n\033[1m### 5. src/evaluate.py ###\033[0m\n")

    params = load_params()
    eval_params = params['evaluate']
    
    # Load test data
    test_data = pd.read_csv(eval_params['test_data'])
    target_col = eval_params['target_column']
    
    print(f"\tUsing target column: {target_col}")
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    print(f"\tTest data shape: {X_test.shape}")
    print(f"\tTest target mean: {y_test.mean():.4f}")
    
    # Load model and scaler
    model = joblib.load(eval_params['model_path'])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save metrics
    metrics = {
        "test_r2": float(r2),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "target_column": target_col
    }
    
    os.makedirs('reports/metrics', exist_ok=True)
    with open(eval_params['metrics_output'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots
    os.makedirs('reports/plots', exist_ok=True)
    
    # Predictions vs. True Values Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Predictions vs. True Values - {target_col}')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.tight_layout()
    plt.savefig('reports/plots/predictions_vs_true_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot (if available)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top 15 Feature Importance - {target_col}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('reports/plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\tTest R²: {r2:.4f}")
    print(f"\tTest RMSE: {rmse:.4f}")
    print(f"\tTest MAE: {mae:.4f}")
    print(f"\tResults saved to: {eval_params['metrics_output']}")

if __name__ == "__main__":
    evaluate_model()
