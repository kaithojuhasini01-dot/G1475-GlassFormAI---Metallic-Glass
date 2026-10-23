import pandas as pd
import os

# Getting the absolute path of the current script's directory (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the main project folder, then into data/raw
project_root = os.path.dirname(current_dir)
raw_file_path = os.path.join(project_root, 'data', 'raw', 'train-00000-of-00001.parquet')

print(f"Looking for file at: {raw_file_path}")

try:
    df = pd.read_parquet(raw_file_path)
    
    print(f"Successfully loaded raw data!")
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    try:
        zeros = df[df['Dmax_[mm]'] == 0]
        print(f"Records with Dmax = 0 (Negative Class): {len(zeros)}")
        
        positives = df[df['Dmax_[mm]'] > 0]
        print(f"Records with Dmax > 0 (Positive Class): {len(positives)}")
        
    except KeyError:
        print("Could not find column 'Dmax_[mm]'. Here are the column names:")
        print(df.columns.tolist())

except Exception as e:
    print(f"Error loading file: {e}")