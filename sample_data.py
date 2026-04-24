import pandas as pd
import os

print("Loading full dataset...")
df = pd.read_csv('crime_dataset.csv')
print(f"Original shape: {df.shape}")

# Calculate fraction to get under 100MB (GitHub limit)
# 250MB -> ~90MB means about 35% of the data
frac = 0.35
df_sample = df.sample(frac=frac, random_state=42)

print(f"Sampled shape: {df_sample.shape}")
df_sample.to_csv('crime_dataset_sample.csv', index=False)
print("Saved crime_dataset_sample.csv!")
print(f"File size: {os.path.getsize('crime_dataset_sample.csv') / (1024*1024):.2f} MB")
