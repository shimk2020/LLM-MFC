# -*- coding: utf-8 -*-
import pandas as pd
import sys

df1 = pd.read_csv(r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\PLS\1 dim\MFC_dataset_embedded_1dim.csv")
try:
    df2 = pd.read_csv(r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\MFC_dataset_imputed.csv", encoding="utf-8")
except:
    df2 = pd.read_csv(r"C:\Users\shimk\Desktop\LLM-BES 코드\머신러닝\MFC_dataset_imputed.csv", encoding="cp949")

print("=== EMBEDDED (1dim) ===")
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print(df1.head(3).to_string())

print("\n\n=== ORIGINAL (imputed) ===")
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
print(df2.dtypes)
print(df2.head(3).to_string())

# Check text columns
for col in df2.columns:
    if df2[col].dtype == 'object':
        print(f"\n--- {col} ---")
        print(f"  Unique values ({df2[col].nunique()}): {df2[col].unique()[:15].tolist()}")
