import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("c:/Users/bhosl/OneDrive/Desktop/loan risk app/database/data/train_LZV4RXX.csv")
print("Columns in dataset:", df.columns.tolist())

# Rename column if needed
if 'education' in df.columns:
    df.rename(columns={'education': 'Education'}, inplace=True)

# Fill missing categorical values with fallback defaults
df['Education'] = df['Education'].fillna('Graduate')
df['proof_submitted'] = df['proof_submitted'].fillna('Yes')
df['last_delinq_none'] = df['last_delinq_none'].fillna('Yes')

# Map categorical values safely
edu_map = {'Graduate': 1, 'Not Graduate': 0}
proof_map = {'Yes': 1, 'No': 0}
delinq_map = {'Yes': 1, 'No': 0}

df['Education'] = df['Education'].map(edu_map)
df['proof_submitted'] = df['proof_submitted'].map(proof_map)
df['last_delinq_none'] = df['last_delinq_none'].map(delinq_map)

# Fill any unmapped values with fallback (in case of typos or unexpected strings)
df['Education'] = df['Education'].fillna(1)
df['proof_submitted'] = df['proof_submitted'].fillna(1)
df['last_delinq_none'] = df['last_delinq_none'].fillna(1)

# Select features
features = ['age', 'Education', 'proof_submitted', 'loan_amount',
            'asset_cost', 'no_of_loans', 'last_delinq_none']
X = df[features].copy()
y = df['loan_default'].copy()

# Fill missing numeric values with median
for col in ['age', 'loan_amount', 'asset_cost', 'no_of_loans']:
    X[col] = X[col].fillna(X[col].median())

# Final check for NaNs
if X.isnull().any().any():
    print("Warning: NaNs still present in X after imputation.")
    print(X.isnull().sum())
    raise ValueError("Cannot proceed: X still contains NaNs.")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_default_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")