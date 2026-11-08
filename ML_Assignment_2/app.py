"""
ML Assignment 2 - BITS Pilani M.Tech (AIML)
Streamlit Application: Multi-Model Classification Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Assignment 2 | BITS Pilani",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    'Logistic Regression':    os.path.join(BASE_DIR, 'model', 'logistic_regression.joblib'),
    'Decision Tree':          os.path.join(BASE_DIR, 'model', 'decision_tree.joblib'),
    'K-Nearest Neighbors':    os.path.join(BASE_DIR, 'model', 'knn.joblib'),
    'Naive Bayes':            os.path.join(BASE_DIR, 'model', 'naive_bayes.joblib'),
    'Support Vector Machine': os.path.join(BASE_DIR, 'model', 'svm.joblib'),
}

SCALER_FILE = os.path.join(BASE_DIR, 'model', 'scaler.joblib')
RESULTS_FILE = os.path.join(BASE_DIR, 'model', 'results.csv')
TARGET_COL = 'income'

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load(SCALER_FILE)

def check_models_exist():
    missing = [f for f in list(MODEL_FILES.values()) + [SCALER_FILE] if not os.path.exists(f)]
    return missing

def compute_metrics(y_true, y_pred, y_prob):
    avg = 'weighted'
    metrics = {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        'F1 Score':  round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y_true, y_pred), 4),
    }
    try:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = round(roc_auc_score(y_true, y_prob), 4)
        else:
            metrics['AUC'] = round(roc_auc_score(y_true, y_prob, multi_class='ovr', average=avg), 4)
    except Exception:
        metrics['AUC'] = 'N/A'
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor='gray'
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_prob, model_name):
    if len(np.unique(y_true)) != 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig

def plot_metrics_bar(metrics_dict, model_name):
    numeric = {k: v for k, v in metrics_dict.items() if isinstance(v, float)}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    bars = ax.bar(numeric.keys(), numeric.values(), color=colors[:len(numeric)], edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, numeric.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(f'Evaluation Metrics — {model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    plt.tight_layout()
    return fig

def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.13
    metric_cols = [c for c in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                   if c in results_df.columns]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    for i, (col, color) in enumerate(zip(metric_cols, colors)):
        vals = pd.to_numeric(results_df[col], errors='coerce')
        offset = (i - len(metric_cols) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=col, color=color, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Comparison — All Metrics', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('Score')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='background: linear-gradient(90deg, #003366, #0055A5); padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0; font-size: 26px;'>🎓 ML Assignment 2 — Classification Dashboard</h1>
    <p style='color: #cce0ff; margin: 4px 0 0 0; font-size: 14px;'>
        BITS Pilani | M.Tech (AIML) | Machine Learning | Work Integrated Learning Programme
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    # Check models exist
    missing = check_models_exist()
    if missing:
        st.error(f"**Models not found!**\n\nPlease run:\n```\npython train_models.py\n```")
        st.stop()
    else:
        st.success("All 5 models loaded ✓")

    # File upload
    st.markdown("#### 1. Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv", type=["csv"],
        help="Upload the test_data.csv generated by train_models.py"
    )

    st.markdown("#### 2. Select Model")
    selected_model_name = st.selectbox(
        "Choose a classifier",
        options=list(MODEL_FILES.keys()),
        index=0
    )

    st.markdown("---")
    st.markdown("**Dataset:** Adult Income (UCI)")
    st.markdown("**Task:** Binary Classification")
    st.markdown("**Target:** Income (>50K / ≤50K)")

# ── Main Content ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload **test_data.csv** from the sidebar to get started.")
    st.markdown("""
    ### How to use this app
    1. Run `python train_models.py` to train all 5 models and generate `test_data.csv`
    2. Upload `test_data.csv` using the sidebar
    3. Select a model from the dropdown
    4. View evaluation metrics, confusion matrix, and ROC curve
    5. Compare all models in the **Model Comparison** tab
    """)

    # Show dataset info
    st.markdown("---")
    st.markdown("### Dataset: Adult Income (UCI)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Features", "14")
    col2.metric("Min Instances", "500+")
    col3.metric("Classes", "2")

    st.markdown("""
    **Features include:** age, workclass, education, marital-status, occupation,
    relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, fnlwgt, education-num

    **Target:** `income` — whether a person earns >50K/year (1) or ≤50K/year (0)
    """)
    st.stop()

# ── Load & validate data ───────────────────────────────────────────────────────
try:
    df_test = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if TARGET_COL not in df_test.columns:
    st.error(f"Target column `{TARGET_COL}` not found in uploaded CSV. "
             f"Available columns: {list(df_test.columns)}")
    st.stop()

X_raw  = df_test.drop(columns=[TARGET_COL])
y_true = df_test[TARGET_COL].values

st.markdown(f"**Uploaded test data:** {df_test.shape[0]} rows × {df_test.shape[1]} columns")

# Scale features
try:
    scaler = load_scaler()
    X_scaled = scaler.transform(X_raw)
except Exception as e:
    st.error(f"Scaling failed: {e}. Make sure the CSV has the same columns as training data.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    f"📊 Selected Model: {selected_model_name}",
    "📋 Classification Report",
    "📈 Model Comparison"
])

# ── Tab 1: Single Model Results ────────────────────────────────────────────────
with tab1:
    st.markdown(f"## {selected_model_name}")
    st.markdown("---")

    # Load & predict
    with st.spinner(f"Running {selected_model_name}..."):
        try:
            model = load_model(MODEL_FILES[selected_model_name])
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Metrics row
    st.markdown("### Evaluation Metrics")
    cols = st.columns(6)
    metric_labels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    icons = ['🎯', '📐', '✅', '🔍', '⚖️', '📏']
    for col, label, icon in zip(cols, metric_labels, icons):
        val = metrics.get(label, 'N/A')
        display = f"{val:.4f}" if isinstance(val, float) else str(val)
        col.metric(f"{icon} {label}", display)

    st.markdown("---")

    # Plots
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred, selected_model_name)
        st.pyplot(fig_cm)
        plt.close()

    with col_right:
        st.markdown("### ROC Curve")
        fig_roc = plot_roc_curve(y_true, y_prob, selected_model_name)
        if fig_roc:
            st.pyplot(fig_roc)
            plt.close()
        else:
            st.info("ROC curve available for binary classification only.")

    st.markdown("### Metrics Bar Chart")
    fig_bar = plot_metrics_bar(metrics, selected_model_name)
    st.pyplot(fig_bar)
    plt.close()

    # Metrics table
    st.markdown("### Metrics Summary Table")
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df.style.format(
        {k: "{:.4f}" for k in metrics_df.columns if metrics_df[k].dtype == float}
    ), use_container_width=True)

# ── Tab 2: Classification Report ──────────────────────────────────────────────
with tab2:
    st.markdown(f"## Classification Report — {selected_model_name}")
    st.markdown("---")

    try:
        model = load_model(MODEL_FILES[selected_model_name])
        y_pred_cr = model.predict(X_scaled)
        report = classification_report(y_true, y_pred_cr, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("### Raw Classification Report")
        st.code(classification_report(y_true, y_pred_cr))
    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 3: All Models Comparison ───────────────────────────────────────────────
with tab3:
    st.markdown("## All Models — Side-by-Side Comparison")
    st.markdown("---")

    all_results = {}
    progress = st.progress(0, text="Evaluating all models...")

    for i, (model_name, model_path) in enumerate(MODEL_FILES.items()):
        try:
            mdl = load_model(model_path)
            yp  = mdl.predict(X_scaled)
            ypr = mdl.predict_proba(X_scaled)[:, 1]
            all_results[model_name] = compute_metrics(y_true, yp, ypr)
        except Exception as e:
            all_results[model_name] = {'Error': str(e)}
        progress.progress((i + 1) / len(MODEL_FILES), text=f"Done: {model_name}")

    progress.empty()

    results_df = pd.DataFrame(all_results).T
    results_df.index.name = 'Model'

    # Color-coded table
    st.markdown("### Comparison Table")
    numeric_cols = [c for c in results_df.columns if results_df[c].dtype in [float, np.float64]]
    styled = (
        results_df[numeric_cols]
        .style
        .format("{:.4f}")
        .background_gradient(cmap='YlGn', axis=0)
        .set_caption("Higher values = better performance (darker = higher)")
    )
    st.dataframe(styled, use_container_width=True)

    # Best model
    if 'Accuracy' in results_df.columns:
        best_model = results_df['Accuracy'].astype(float).idxmax()
        best_acc   = results_df.loc[best_model, 'Accuracy']
        st.success(f"🏆 **Best Model by Accuracy:** {best_model} — `{best_acc:.4f}`")

    if 'F1 Score' in results_df.columns:
        best_f1_model = results_df['F1 Score'].astype(float).idxmax()
        best_f1       = results_df.loc[best_f1_model, 'F1 Score']
        st.info(f"⚖️ **Best Model by F1 Score:** {best_f1_model} — `{best_f1:.4f}`")

    # Comparison bar chart
    st.markdown("### Metrics Comparison Chart")
    fig_cmp = plot_model_comparison(results_df)
    st.pyplot(fig_cmp)
    plt.close()

    # Download results
    csv_out = results_df.to_csv().encode('utf-8')
    st.download_button(
        label="⬇️ Download Comparison Table as CSV",
        data=csv_out,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

    # Observations section
    st.markdown("---")
    st.markdown("### Model Observations")
    observations = {
        'Logistic Regression': (
            "Performs well on linearly separable data. Provides good baseline accuracy with fast training. "
            "Works best when features are scaled and relationships are approximately linear. "
            "Tends to underfit complex non-linear patterns in the Adult Income dataset."
        ),
        'Decision Tree': (
            "Captures non-linear patterns effectively without feature scaling. "
            "Prone to overfitting without depth constraints; max_depth=10 was applied. "
            "Highly interpretable model — decision paths can be visualized directly. "
            "Performance may lag behind ensemble methods on this dataset."
        ),
        'K-Nearest Neighbors': (
            "Non-parametric model sensitive to feature scaling (StandardScaler applied). "
            "Performance depends heavily on the choice of k; k=7 used here. "
            "Computationally expensive at prediction time for large datasets. "
            "Generally achieves moderate accuracy on tabular data like Adult Income."
        ),
        'Naive Bayes': (
            "Fastest training and prediction among all models. "
            "Assumes feature independence (Gaussian NB used), which is violated in practice. "
            "Despite the independence assumption, achieves reasonable accuracy. "
            "Best suited as a fast baseline rather than a high-performance classifier here."
        ),
        'Support Vector Machine': (
            "RBF kernel captures complex non-linear decision boundaries effectively. "
            "Typically achieves competitive accuracy on tabular classification tasks. "
            "Training is computationally expensive (O(n²–n³)), hence data was sampled. "
            "Often among the top performers on the Adult Income dataset with proper tuning."
        )
    }

    for model_name, obs in observations.items():
        with st.expander(f"**{model_name}**"):
            st.write(obs)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 13px;'>
    BITS Pilani — M.Tech (AIML) | Machine Learning | Assignment 2<br>
    Dataset: Adult Income (UCI ML Repository) | Models: LR · DT · KNN · NB · SVM
</div>
""", unsafe_allow_html=True)
