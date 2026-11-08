"""
ML Assignment 2 - BITS Pilani M.Tech (AIML)
Training Script: 5 Classification Models on Adult Income Dataset (UCI)

Run this script FIRST to train all models and generate:
  - model/*.joblib  (trained models + scaler)
  - test_data.csv   (for Streamlit upload)
  - model/results.csv (metrics comparison)
"""

import pandas as pd
import numpy as np
import os
import ssl
import warnings
warnings.filterwarnings('ignore')

# Bypass SSL verification for enterprise/corporate networks with SSL inspection
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 60)
print("  ML Assignment 2 — Model Training")
print("  BITS Pilani M.Tech (AIML)")
print("=" * 60)
print("\n[1/5] Loading Adult Income Dataset from UCI...")

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

try:
    df = pd.read_csv(URL, names=COLUMNS, sep=r',\s*', engine='python', na_values='?')
    print(f"    Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
except Exception as e:
    print(f"    Network error: {e}")
    print("    Please manually download 'adult.data' from UCI and place it in this folder.")
    print("    Then rename it to 'adult.data' and re-run this script.")
    LOCAL = 'adult.data'
    if os.path.exists(LOCAL):
        df = pd.read_csv(LOCAL, names=COLUMNS, sep=r',\s*', engine='python', na_values='?')
        print(f"    Loaded from local file: {df.shape}")
    else:
        raise FileNotFoundError("Dataset not found. Download from UCI first.")

# ── 2. Preprocess ──────────────────────────────────────────────────────────────
print("\n[2/5] Preprocessing...")

df.dropna(inplace=True)
print(f"    After removing missing values: {df.shape[0]} rows")

# Encode binary target: <=50K -> 0, >50K -> 1
df['income'] = (df['income'].str.strip() == '>50K').astype(int)
print(f"    Class distribution: {df['income'].value_counts().to_dict()}")

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"    Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")

# Features and target
X = df.drop('income', axis=1)
y = df['income']
print(f"    Features: {X.shape[1]}  |  Instances: {X.shape[0]}")

# Sample 10,000 rows for efficiency (SVM training speed)
if len(df) > 10000:
    sample_idx = df.sample(n=10000, random_state=42).index
    X = X.loc[sample_idx]
    y = y.loc[sample_idx]
    print(f"    Sampled to 10,000 rows for efficient SVM training.")

# ── 3. Train / Test Split ──────────────────────────────────────────────────────
print("\n[3/5] Splitting data (75% train / 25% test)...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"    Train: {X_train_raw.shape[0]}  |  Test: {X_test_raw.shape[0]}")

# Save unscaled test data (for Streamlit upload)
test_data = X_test_raw.copy()
test_data['income'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print(f"    Saved test_data.csv ({test_data.shape[0]} rows)")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Save preprocessing artifacts
os.makedirs('model', exist_ok=True)
joblib.dump(scaler,          'model/scaler.joblib')
joblib.dump(label_encoders,  'model/label_encoders.joblib')
print("    Saved scaler.joblib and label_encoders.joblib")

# ── 4. Train & Evaluate All 5 Models ──────────────────────────────────────────
print("\n[4/5] Training models...")

MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'knn':                 KNeighborsClassifier(n_neighbors=7),
    'naive_bayes':         GaussianNB(),
    'svm':                 SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'knn':                 'K-Nearest Neighbors',
    'naive_bayes':         'Naive Bayes',
    'svm':                 'Support Vector Machine'
}

results = {}

for key, model in MODELS.items():
    display = DISPLAY_NAMES[key]
    print(f"\n    Training {display}...", end='', flush=True)
    model.fit(X_train, y_train)
    joblib.dump(model, f'model/{key}.joblib')

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'AUC':       round(roc_auc_score(y_test, y_prob), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall':    round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1':        round(f1_score(y_test, y_pred, average='weighted'), 4),
        'MCC':       round(matthews_corrcoef(y_test, y_pred), 4)
    }
    results[display] = metrics

    print(f" Done.")
    print(f"      Acc={metrics['Accuracy']}  AUC={metrics['AUC']}  "
          f"F1={metrics['F1']}  MCC={metrics['MCC']}")

# ── 5. Save Results & Print Summary ───────────────────────────────────────────
print("\n[5/5] Saving results...")

results_df = pd.DataFrame(results).T
results_df.to_csv('model/results.csv')

print("\n" + "=" * 60)
print("  MODEL COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string())

print("\n" + "=" * 60)
print("  ALL FILES SAVED")
print("=" * 60)
print("  model/logistic_regression.joblib")
print("  model/decision_tree.joblib")
print("  model/knn.joblib")
print("  model/naive_bayes.joblib")
print("  model/svm.joblib")
print("  model/scaler.joblib")
print("  model/label_encoders.joblib")
print("  model/results.csv")
print("  test_data.csv")
print("\nTraining complete! Now run: streamlit run app.py")
print("=" * 60)
