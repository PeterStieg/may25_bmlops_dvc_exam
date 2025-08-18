# src/evaluate.py
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model():
    params = load_params()
    eval_params = params['evaluate']
    
    # Load test data
    test_data = pd.read_csv(eval_params['test_data'])
    target_col = test_data.columns[-1]  # Assume last column is target
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Load model
    model = joblib.load(eval_params['model_path'])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(report['macro avg']['precision']),
        "recall": float(report['macro avg']['recall']),
        "f1_score": float(report['macro avg']['f1-score'])
    }
    
    os.makedirs('reports/metrics', exist_ok=True)
    with open(eval_params['metrics_output'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots
    os.makedirs('reports/plots', exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/plots/confusion_matrix.png')
    plt.close()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance_df)), importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45)
        plt.tight_layout()
        plt.savefig('reports/plots/feature_importance.png')
        plt.close()
    
    print(f"Evaluation complete!")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Results saved to: {eval_params['metrics_output']}")

if __name__ == "__main__":
    evaluate_model()
