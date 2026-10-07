"""
train.py
--------
Complete training pipeline for the Financial Fraud Detection project.

Pipeline:
    Load Dataset -> Clean -> Split -> Scale -> SMOTE -> Train Models
    -> Evaluate -> Compare -> Select Best -> Save Model/Scaler/Metrics

Run with:
    python train.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration / paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "creditcard.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
EDA_PATH = os.path.join(MODELS_DIR, "eda_summary.json")
SAMPLES_PATH = os.path.join(MODELS_DIR, "sample_transactions.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest size control. Without a depth limit every tree memorises the
# SMOTE-balanced data and the saved model can reach 700+ MB, which GitHub and
# Streamlit Cloud will not accept. A tree of depth 12 has at most 8,191 nodes,
# so 100 trees stay far below GitHub's 100 MB file limit.
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 12
MAX_MODEL_MB = 90


# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------
def load_dataset(path):
    """Load the credit card transactions dataset from disk."""
    print("Loading dataset...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud) and place it in the data/ folder."
        )

    df = pd.read_csv(path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Clean dataset
# ---------------------------------------------------------------------------
def clean_dataset(df):
    """Remove duplicate rows and report basic dataset info."""
    print("Cleaning dataset...")

    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Removed {n_before - n_after} duplicate rows.")

    n_missing = df.isnull().sum().sum()
    print(f"Missing values found: {n_missing}")
    if n_missing > 0:
        df = df.dropna()
        print("Rows with missing values dropped.")

    return df


# ---------------------------------------------------------------------------
# Step 3: Quick statistics (mirrors the exploratory analysis)
# ---------------------------------------------------------------------------
def print_class_distribution(y, label=""):
    counts = y.value_counts().sort_index()
    total = len(y)
    genuine = counts.get(0, 0)
    fraud = counts.get(1, 0)
    fraud_pct = (fraud / total) * 100 if total > 0 else 0
    print(f"{label} -> Genuine: {genuine} | Fraud: {fraud} "
          f"| Fraud rate: {fraud_pct:.4f}%")


def show_dataset_summary(df):
    print("\n--- Dataset Summary ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print_class_distribution(df["Class"], label="Overall")
    print("------------------------\n")


# ---------------------------------------------------------------------------
# Step 4: Split features/target and train/test split
# ---------------------------------------------------------------------------
def split_data(df):
    """Separate features/target and create a stratified train/test split."""
    print("Splitting data into train and test sets...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print_class_distribution(y_train, label="Train (before SMOTE)")
    print_class_distribution(y_test, label="Test (untouched)")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 5: Scale features
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    """
    Fit a StandardScaler on the Amount column of the TRAINING data only,
    then use the fitted scaler to transform both train and test data.
    This avoids data leakage from the test set into preprocessing.
    """
    print("Scaling 'Amount' feature (fit on training data only)...")

    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    # Transform test data using the already-fitted scaler
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Step 6: Handle class imbalance with SMOTE (training data only)
# ---------------------------------------------------------------------------
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data only. The test set must remain
    untouched so that evaluation reflects real-world class imbalance.
    """
    print("Applying SMOTE to training data...")
    print_class_distribution(y_train, label="Before SMOTE")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print_class_distribution(y_resampled, label="After SMOTE")

    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# Step 7: Train models
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train):
    """
    Isolation Forest is an unsupervised anomaly-detection model.
    It is trained WITHOUT labels, on the original (non-SMOTE) training
    data, since SMOTE-generated synthetic samples would distort the
    notion of what a 'normal' transaction looks like.
    """
    print("Training Isolation Forest...")
    model = IsolationForest(
        contamination=0.0017,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    return model


# ---------------------------------------------------------------------------
# Step 8: Evaluate models
# ---------------------------------------------------------------------------

def downsample_curve(x, y, max_points=200):
    """Keep curve files small: at most `max_points` evenly spaced points."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return [round(float(v), 5) for v in x], [round(float(v), 5) for v in y]


def compute_metrics(y_test, y_pred, y_scores):
    """Precision / Recall / F1 / PR-AUC / ROC-AUC + confusion matrix + curves."""
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_scores)
    fpr_c, tpr_c, _ = roc_curve(y_test, y_scores)
    pr_x, pr_y = downsample_curve(rec_c, prec_c)
    roc_x, roc_y = downsample_curve(fpr_c, tpr_c)

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_scores),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "pr_curve": {"recall": pr_x, "precision": pr_y},
        "roc_curve": {"fpr": roc_x, "tpr": roc_y},
    }


def print_metrics(name, m):
    print(f"\n{name} results:")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1-score:  {m['f1_score']:.4f}")
    print(f"  PR-AUC:    {m['pr_auc']:.4f}")


def evaluate_classifier(model, X_test, y_test, name):
    """Evaluate a standard sklearn classifier on the untouched test set."""
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, y_pred, y_scores)
    print_metrics(name, m)
    return m


def evaluate_isolation_forest(model, X_test, y_test):
    """
    Isolation Forest predicts -1 for anomalies (fraud) and 1 for normal
    points. We map these to the same 0/1 convention used elsewhere
    (1 = fraud) so metrics are comparable.
    """
    raw_pred = model.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)

    # decision_function: lower score = more abnormal. Invert so that
    # higher score = higher fraud likelihood, for PR-AUC purposes.
    y_scores = -model.decision_function(X_test)

    m = compute_metrics(y_test, y_pred, y_scores)
    print_metrics("Isolation Forest", m)
    return m


# ---------------------------------------------------------------------------
# Step 9: Select the best model
# ---------------------------------------------------------------------------
def select_best_model(results):
    """
    Select the best classification model using F1-score as the primary
    ranking metric (a balance of precision and recall), which is more
    meaningful than accuracy for a highly imbalanced fraud dataset.
    Isolation Forest is reported for comparison but, being an
    unsupervised method, is not selected as the deployed model.
    """
    candidates = {k: v for k, v in results.items() if k != "Isolation Forest"}
    best_name = max(candidates, key=lambda k: candidates[k]["f1_score"])
    return best_name


# ---------------------------------------------------------------------------
# Extra outputs so the web app does NOT need the 150 MB dataset at runtime
# ---------------------------------------------------------------------------
def histogram_dict(values, bins=40, upper_quantile=None):
    """Return histogram counts/edges as plain lists (JSON friendly)."""
    values = np.asarray(values)
    if upper_quantile is not None:
        values = values[values <= np.quantile(values, upper_quantile)]
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges]}


def build_eda_summary(df, n_duplicates_removed):
    """Pre-compute everything the Analytics page needs."""
    fraud = df[df["Class"] == 1]
    genuine = df[df["Class"] == 0]

    corr = df.corr()["Class"].drop("Class")
    top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)

    return {
        "total": int(len(df)),
        "fraud": int(len(fraud)),
        "genuine": int(len(genuine)),
        "duplicates_removed": int(n_duplicates_removed),
        "amount_stats": {
            "genuine_mean": round(float(genuine["Amount"].mean()), 2),
            "genuine_median": round(float(genuine["Amount"].median()), 2),
            "fraud_mean": round(float(fraud["Amount"].mean()), 2),
            "fraud_median": round(float(fraud["Amount"].median()), 2),
            "fraud_max": round(float(fraud["Amount"].max()), 2),
        },
        # Amount histograms are clipped at the 99th percentile so a few
        # huge outliers do not squash the whole chart.
        "amount_hist_all": histogram_dict(df["Amount"], upper_quantile=0.99),
        "amount_hist_fraud": histogram_dict(fraud["Amount"], upper_quantile=0.99),
        "time_hist_genuine": histogram_dict(genuine["Time"] / 3600, bins=48),
        "time_hist_fraud": histogram_dict(fraud["Time"] / 3600, bins=48),
        "top_correlations": {k: round(float(v), 4) for k, v in top_corr.items()},
    }


def get_feature_importance(model, feature_names, top_n=15):
    """Tree models expose feature_importances_, linear models coef_."""
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Importance (Gini)"
    elif hasattr(model, "coef_"):
        values, label = np.abs(model.coef_[0]), "|Coefficient|"
    else:
        return None
    order = np.argsort(values)[::-1][:top_n]
    return {
        "label": label,
        "features": [feature_names[i] for i in order],
        "values": [round(float(values[i]), 5) for i in order],
    }


def save_sample_transactions(model, X_test_raw, X_test_scaled, y_test, n_each=5):
    """
    Save a few REAL (unscaled) test transactions for the app's
    "load example" buttons, so nobody has to type 30 numbers by hand.
    Only examples the deployed model classifies correctly are kept, so
    the demo is easy to follow (the app says this openly).
    """
    probs = model.predict_proba(X_test_scaled)[:, 1]
    correct = (probs >= 0.5).astype(int) == y_test.values

    samples = X_test_raw.copy()
    samples["Class"] = y_test.values
    samples = samples[correct]

    fraud = samples[samples["Class"] == 1]
    genuine = samples[samples["Class"] == 0]
    fraud = fraud.sample(min(n_each, len(fraud)), random_state=RANDOM_STATE)
    genuine = genuine.sample(min(n_each, len(genuine)), random_state=RANDOM_STATE)
    pd.concat([genuine, fraud]).to_csv(SAMPLES_PATH, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1-2. Load and clean
    df = load_dataset(DATA_PATH)
    rows_before = len(df)
    df = clean_dataset(df)
    show_dataset_summary(df)

    # Summary used by the Analytics page of the web app
    with open(EDA_PATH, "w") as f:
        json.dump(build_eda_summary(df, rows_before - len(df)), f, indent=2)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Scale (fit on train only)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 5. SMOTE (train only)
    X_train_resampled, y_train_resampled = apply_smote(
        X_train_scaled, y_train
    )

    # 6. Train models
    log_reg = train_logistic_regression(X_train_resampled, y_train_resampled)
    rand_forest = train_random_forest(X_train_resampled, y_train_resampled)
    # Isolation Forest trained on original (non-resampled) scaled data
    iso_forest = train_isolation_forest(X_train_scaled)

    # 7. Evaluate on the untouched, original-distribution test set
    print("\nEvaluating models on the untouched test set...")
    results = {
        "Logistic Regression": evaluate_classifier(
            log_reg, X_test_scaled, y_test, "Logistic Regression"
        ),
        "Random Forest": evaluate_classifier(
            rand_forest, X_test_scaled, y_test, "Random Forest"
        ),
        "Isolation Forest": evaluate_isolation_forest(
            iso_forest, X_test_scaled, y_test
        ),
    }

    # 8. Comparison table
    print("\n--- Model Comparison ---")
    print(f"{'Model':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'PR-AUC':>10}")
    for name, r in results.items():
        print(f"{name:<22}{r['precision']:>10.4f}{r['recall']:>10.4f}"
              f"{r['f1_score']:>10.4f}{r['pr_auc']:>10.4f}")

    # 9. Select best classification model
    best_name = select_best_model(results)
    print(f"\nBest model: {best_name}")

    best_model = {
        "Logistic Regression": log_reg,
        "Random Forest": rand_forest,
    }[best_name]

    # 10. Save model, scaler, metrics
    print("Saving model...")
    joblib.dump(best_model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH)
    save_sample_transactions(best_model, X_test, X_test_scaled, y_test)

    feature_names = list(X_train.columns)
    metrics_output = {
        "best_model": best_name,
        "feature_order": feature_names,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_importance": get_feature_importance(best_model, feature_names),
        "results": results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model file size: {size_mb:.1f} MB")
    if size_mb > MAX_MODEL_MB:
        print(f"WARNING: model is larger than {MAX_MODEL_MB} MB. GitHub rejects files over "
              f"100 MB. Lower RF_MAX_DEPTH (e.g. 10) or RF_N_ESTIMATORS (e.g. 60) at the "
              f"top of train.py and run it again.")

    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"EDA summary saved to: {EDA_PATH}")
    print(f"Sample transactions saved to: {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
