import pandas as pd
import numpy as np
from pathlib import Path

# The 14 features of the dataset used from the original attributes
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'num'
]

# Path to the data folder (goes up one level from src/, into data/)
DATA_DIR = Path(__file__).parent.parent / 'data'

def load_data():
    files = {
        'cleveland':    'processed.cleveland.data',
        'hungarian':    'processed.hungarian.data',
        'switzerland':  'processed.switzerland.data',
        'va':           'processed.va.data',
    }
    
    dfs = []
    for source, filename in files.items():
        df = pd.read_csv(
            DATA_DIR / filename,
            header=None,        # no header row in these files
            names=COLUMNS,      # assign our column names
            na_values=['?']     # treat '?' as missing values (NaN)
        )
        df['source'] = source   # track which hospital it came from
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def preprocess (df):
    # Binarize the target: 0 = healthy, 1 = disease (was 0-4)
    df['target'] = (df['num']>0).astype(int)
    df = df.drop(columns=['num'])
    
    # Fill missing values with the median of each column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def main():
    print("loading dara...")
    df = load_data()
    print(f" Loaded {len(df)} rows from 4 sources")
    
    print("Preprocessing...")
    df = preprocess(df)
    print(f" Done! Shape: {df.shape}")
    print(f" Disease prevalence: {df['target'].mean()*100:.1f}%")
    
    # Save cleaned data to data/ folder
    output_path = DATA_DIR / 'heart_disease_clean.csv'
    df.to_csv(output_path, index=False)
    print(f" Saved to {output_path}")
    
if __name__ == "__main__":
    main()
    