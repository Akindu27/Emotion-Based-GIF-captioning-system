import pandas as pd
import numpy as np

# Load the full dataset
df = pd.read_csv('gifgif-dataset-20150121-v1.csv')

print("=" * 60)
print("GIFGIF Dataset Overview")
print("=" * 60)

print(f"\n📊 Total comparisons: {len(df):,}")
print(f"📁 File size: 114.63 MB")
print(f"📋 Columns: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("Emotion Distribution")
print("=" * 60)
print(df['metric'].value_counts())

print("\n" + "=" * 60)
print("Choice Distribution")
print("=" * 60)
print(df['choice'].value_counts())

print("\n" + "=" * 60)
print("Unique GIFs")
print("=" * 60)
unique_gifs = set(df['left'].unique()) | set(df['right'].unique())
print(f"Total unique GIFs: {len(unique_gifs):,}")

print("\n" + "=" * 60)
print("Sample comparisons:")
print("=" * 60)
print(df.head(10))