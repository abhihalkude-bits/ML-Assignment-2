# ML Assignment 2 — Classification Dashboard

**BITS Pilani | M.Tech (AIML) | Machine Learning | Work Integrated Learning Programme**

---

## a. Problem Statement

Predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes. This is a **binary classification** task where the goal is to compare the performance of five different ML classifiers on the same dataset and deploy an interactive evaluation dashboard.

---

## b. Dataset Description

| Attribute          | Details                                                  |
|--------------------|----------------------------------------------------------|
| **Name**           | Adult Income Dataset (Census Income)                     |
| **Source**         | UCI Machine Learning Repository                          |
| **Task**           | Binary Classification                                    |
| **Target**         | `income` — `>50K` (1) or `≤50K` (0)                    |
| **Features**       | 14 (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) |
| **Instances**      | ~48,842 (after removing missing values: ~45,222)         |
| **Class Balance**  | ~76% ≤50K, ~24% >50K (imbalanced)                       |
| **Feature Types**  | Mix of numeric and categorical                           |

**Preprocessing steps applied:**
- Removed rows with missing values (`?`)
- Label-encoded all categorical features
- Applied `StandardScaler` for models sensitive to scale (LR, KNN, SVM)
- Stratified 75/25 train/test split with `random_state=42`

---

## c. GitHub Repository Link

> **https://github.com/abhihalkude-bits/ML-Assignment-2**

Repository contains:
- `app.py` — Streamlit application
- `train_models.py` — Model training script
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `test_data.csv` — Test split used for evaluation
- `model/` — Saved `.joblib` files for all 5 classifiers + scaler

---

## d. Models Used

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.8252   | 0.8600 | 0.8155    | 0.8252 | 0.8134 | 0.4971 |
| Decision Tree          | 0.8388   | 0.8600 | 0.8339    | 0.8388 | 0.8355 | 0.5580 |
| K-Nearest Neighbors    | 0.8172   | 0.8532 | 0.8120    | 0.8172 | 0.8140 | 0.5007 |
| Naive Bayes            | 0.8096   | 0.8557 | 0.7993    | 0.8096 | 0.7868 | 0.4330 |
| Support Vector Machine | 0.8404   | 0.8851 | 0.8330    | 0.8404 | 0.8322 | 0.5479 |

---

### Model Observations

| ML Model Name          | Observation about model performance |
|------------------------|--------------------------------------|
| **Logistic Regression**    | Provides a strong linear baseline. Performs well due to many near-linear feature relationships in the Adult Income dataset. Scaling is essential. Tends to underfit complex non-linear decision boundaries. |
| **Decision Tree**          | Captures non-linear patterns without scaling. Depth-limited (max_depth=10) to avoid overfitting. Highly interpretable. Slightly lower performance than SVM/LR on this dataset due to greedy splitting. |
| **K-Nearest Neighbors**    | Moderate performance. Sensitive to scaling and choice of k. Slower prediction time with large test sets. Works best with k=7 on this dataset; however, class imbalance affects minority class recall. |
| **Naive Bayes**            | Fastest model. Feature independence assumption is violated (features are correlated), leading to lower accuracy compared to others. Useful as a quick probabilistic baseline. |
| **Support Vector Machine** | Typically the strongest performer with RBF kernel. Captures complex boundaries. Computationally expensive but achieves competitive accuracy and AUC, especially with imbalanced data. |
| **Overall Winner?**        | **Support Vector Machine** — generally achieves the best balance of accuracy, AUC, and F1 on the Adult Income dataset when properly scaled. |

---

## Live Streamlit App

> **https://abhi-halkude-ml-assignment-2.streamlit.app/**

---

## Repository Structure

```
ml-assignment-2-aiml/
├── app.py                  # Streamlit application
├── train_models.py         # Model training & evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test split CSV (for Streamlit upload)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── scaler.joblib
    ├── label_encoders.joblib
    └── results.csv
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-assignment-2-aiml.git
cd ml-assignment-2-aiml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates model/*.joblib and test_data.csv)
python train_models.py

# 4. Launch Streamlit app
streamlit run app.py
```

Then open `http://localhost:8501` and upload `test_data.csv` to evaluate models.

---

## Streamlit App Features

| Feature | Description |
|---------|-------------|
| CSV Upload | Upload test_data.csv to evaluate pre-trained models |
| Model Dropdown | Select any of the 5 classifiers |
| Evaluation Metrics | Accuracy, AUC, Precision, Recall, F1, MCC |
| Confusion Matrix | Heatmap of true vs predicted labels |
| ROC Curve | AUC visualization for binary classification |
| Model Comparison | Side-by-side bar chart and color-coded table |
| Download Results | Export comparison table as CSV |

---

*BITS Pilani — Work Integrated Learning Programmes Division*
