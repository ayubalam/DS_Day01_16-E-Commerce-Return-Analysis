import pandas as pd

input_path = "data/raw/train.csv"
output_path = "data/processed/cleaned_returns.csv"

df = pd.read_csv(input_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDataset Information:")
df.info()

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)
print("Duplicates After Cleaning:", df.duplicated().sum())

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)