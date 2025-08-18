import pandas as pd
import yaml
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_data():

    # Print this file's name in bold formatting using ASCII
    print("\n\033[1m### 1. src/prepare.py ###\033[0m\n")

    params = load_params()
    
    # Download data from the provided URL
    data_url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv" 
    raw_data = params['prepare']['input']
    output_path = params['prepare']['output']
    
    print(f"\tDownloading data from: {data_url}")
    data = pd.read_csv(data_url)
    data.to_csv(raw_data, index=False)
    
    print(f"\tData shape (raw): {data.shape}")
    print(f"\tColumns (raw): {list(data.columns)}")
    print(f"\tRaw data saved to: {raw_data}")
    
    # Basic data cleaning
    data = data.dropna()  # Remove missing values
    data = data.drop_duplicates()  # Remove duplicates
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"\tData shape (processed): {data.shape}")
    print(f"\tProcessed data saved to: {output_path}")

if __name__ == "__main__":
    prepare_data()
