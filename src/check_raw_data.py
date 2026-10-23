import pandas as pd

raw_file_path = 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\raw\\train-00000-of-00001.parquet' 

try:
    # Parquet files require a special engine, pandas handles it automatically
    df = pd.read_parquet(raw_file_path)
    
    print(f"Successfully loaded raw data!")
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Checking the Dmax column
    # We use a try/except just in case the column name is slightly different
    try:
        # Counting how many have Dmax = 0
        zeros = df[df['Dmax_[mm]'] == 0]
        print(f"Records with Dmax = 0 (Negative Class): {len(zeros)}")
        
        # Counting how many have Dmax > 0
        positives = df[df['Dmax_[mm]'] > 0]
        print(f"Records with Dmax > 0 (Positive Class): {len(positives)}")
        
    except KeyError:
        print("Could not find column 'Dmax_[mm]'. Here are the column names:")
        print(df.columns.tolist())

except Exception as e:
    print(f"Error loading file: {e}")
    print("Check if the file path and name are correct.")