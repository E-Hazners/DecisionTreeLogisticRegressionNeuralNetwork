# PART 1 Data Vectorization and Preparation for Classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1.2 Load dataset and convert to numpy array
CSV_PATH = "STxxxx9.csv"

df = pd.read_csv(CSV_PATH)


print("1.2 Raw DataFrame (first 3 rows):")
print(df.head(3))
print()


# 1.3 Dimension information

print("1.3  Array dimensions")
print(f"Rows (observations): {df.shape[0]}")
print(f"Columns (variables): {df.shape[1]}")
print(f"Column names: {df.columns.tolist()}")
print()

# 1.4 Drop rows with missing values
before = len(df)
df = df.dropna()
after  = len(df)

print("1.4 Drop rows with empty / NaN values")
print(f"Rows before: {before}")
print(f"Rows after: {after}")
print(f"Removed: {before - after}")
print()


# 1.5 One-Hot Encoding for the categorical target column 'Class'

print("1.5 One-Hot Encoding of categorical column 'Class'")

# Identify categorical columns
cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print(f"Categorical columns found: {cat_cols}")

# Apply pd.get_dummies (one-hot encoding), drop first to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

print(f"Columns after encoding: {df_encoded.columns.tolist()}")
print()

# Build feature matrix X and integer-encoded target y (for sklearn)
feature_cols = [c for c in df_encoded.columns
                if not c.startswith("Class_")]
X_df = df_encoded[feature_cols]

le = LabelEncoder()
y   = le.fit_transform(df["Class"])   # integer labels 0-4
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Convert to numpy arrays
X = X_df.to_numpy(dtype=float)
print(f"Feature matrix X shape : {X.shape}")
print(f"Target vector  y shape : {y.shape}")
print()

# 1.6 First 5 observations (encoded)
print("1.6 First 5 observations (feature matrix, encoded):")
print(pd.DataFrame(X[:5], columns=X_df.columns).to_string(index=False))
print()
print("Corresponding class labels (integer):", y[:5])
print("Corresponding class labels (original):", le.inverse_transform(y[:5]))
print()

# 1.7 Split into 75 % / 25 % subsamples
X_75, X_25, y_75, y_25 = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y          # keep class distribution balanced
)

print("1.7 - 75 % / 25 % split")
print(f"Subsample 1 (75 %) - X shape: {X_75.shape}, y shape: {y_75.shape}")
print(f"Subsample 2 (25 %) - X shape: {X_25.shape}, y shape: {y_25.shape}")
print()

# Export for use by other parts
np.save("X_data.npy", X)
np.save("y_data.npy", y)
np.save("feature_names.npy", np.array(X_df.columns.tolist()))
np.save("class_names.npy",   le.classes_)

print("Saved: X_data.npy, y_data.npy, feature_names.npy, class_names.npy")