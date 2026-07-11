import pandas as pd
import os

print("Initializing dynamic data structures...")

data_dir = "data"

# 1. Convert Matriks_Biner_Transaksi.xlsx to Parquet
matrix_excel = os.path.join(data_dir, "Matriks_Biner_Transaksi.xlsx")
matrix_parquet = os.path.join(data_dir, "Matriks_Biner_Transaksi.parquet")

if not os.path.exists(matrix_parquet):
    print(f"Reading {matrix_excel} (this may take a minute)...")
    df_matrix = pd.read_excel(matrix_excel)
    print("Converting to boolean just in case...")
    df_matrix = df_matrix.astype(bool)
    print(f"Saving to {matrix_parquet}...")
    df_matrix.to_parquet(matrix_parquet, index=False)
else:
    print(f"{matrix_parquet} already exists.")

# 2. Copy the rules to a dynamic CSV
rules_excel = os.path.join(data_dir, "Rekomendasi_CrossSell.xlsx")
rules_csv = os.path.join(data_dir, "dynamic_rules.csv")

if not os.path.exists(rules_csv):
    print(f"Reading {rules_excel}...")
    df_rules = pd.read_excel(rules_excel)
    
    # Clean up sets if any
    def clean_item(x):
        if isinstance(x, str):
            x = x.replace("frozenset({", "").replace("})", "")
            x = x.replace("(", "").replace(")", "")
            x = x.replace("'", "").replace('"', "")
            return x.strip()
        return str(x)
        
    df_rules['antecedents'] = df_rules['antecedents'].apply(clean_item)
    df_rules['consequents'] = df_rules['consequents'].apply(clean_item)
    
    print(f"Saving to {rules_csv}...")
    df_rules.to_csv(rules_csv, index=False)
else:
    print(f"{rules_csv} already exists.")

print("Initialization complete!")
