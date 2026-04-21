from app import load_fred_data
import os
from dotenv import load_dotenv

load_dotenv()
df1 = load_fred_data("UNRATE")
df2 = load_fred_data("FEDFUNDS")

print(f"df1 length: {len(df1)}")
print(f"df2 length: {len(df2)}")

df_combined = df1.join(df2, how="inner").dropna()
print(f"combined inner length: {len(df_combined)}")
