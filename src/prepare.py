import pandas as pd
import yaml
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_data():
    params = load_params()
    
    # Download data from the provided URL
    data_url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    output_path = params['prepare']['output']
    
    print(f"Downloading data from: {data_url}")
    data = pd.read_csv(data_url)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Basic data cleaning
    data = data.dropna()  # Remove missing values
    data = data.drop_duplicates()  # Remove duplicates
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print(f"Final shape: {data.shape}")

if __name__ == "__main__":
    prepare_data()
    