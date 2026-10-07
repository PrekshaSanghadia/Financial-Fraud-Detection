# Financial Fraud Detection Using Machine Learning

A B.Tech Semester 5 "Python for Data Science" PBL project that detects
fraudulent credit card transactions using machine learning, with a
Streamlit web application for interactive and batch predictions.

## Description

Credit card fraud is rare but costly — in the dataset used here, only
about 0.17% of transactions are fraudulent. This project builds a
complete pipeline that cleans the data, explores it, handles the
severe class imbalance with SMOTE, trains and compares multiple
models, and deploys the best one behind a simple web interface.

## Features

- Single-transaction check with fraud probability, risk level and one-click real examples
- Adjustable decision threshold to trade missed fraud against false alarms
- Batch CSV upload with a summary, a "flagged only" filter and downloadable results
- Analytics page: class balance, amount and time patterns, features most related to fraud
- Model Performance page: confusion matrix, precision-recall curves, model comparison, feature importance
- Works on Streamlit Cloud without the large dataset (training saves everything the app needs)
- Custom "Fraud Desk" theme: bundled fonts, stamped FLAGGED / CLEARED verdicts and a risk meter

## Technologies

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- imbalanced-learn (SMOTE)
- Joblib
- Streamlit

## Dataset

[Credit Card Fraud Detection Dataset (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it at:

```
data/creditcard.csv
```

The dataset is not included in this repository (see `.gitignore`).

## Installation

```bash
python -m venv venv
```

Activate the environment (Windows):

```bash
venv\Scripts\activate
```

Activate the environment (macOS/Linux):

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the models and generate everything the app needs
(`fraud_model.pkl`, `scaler.pkl`, `metrics.json`, `eda_summary.json`,
`sample_transactions.csv`) inside `models/`:

```bash
python train.py
```

## Run Application

```bash
streamlit run app.py
```

## Project Structure

```
Financial-Fraud-Detection/
│
├── data/
│   └── creditcard.csv        # downloaded separately from Kaggle
│
├── models/
│   ├── fraud_model.pkl       # created by train.py
│   ├── scaler.pkl            # created by train.py
│   ├── metrics.json          # metrics, curves, feature importance
│   ├── eda_summary.json      # numbers behind the Analytics page
│   └── sample_transactions.csv  # examples for the app's demo buttons
│
├── notebooks/
│   └── analysis.ipynb        # exploratory data analysis notebook
│
├── static/fonts/             # bundled fonts (Hanken Grotesk, Source Serif)
├── .streamlit/config.toml    # theme, colours and font settings
│
├── app.py                    # Streamlit application ("Fraud Desk")
├── train.py                  # full training pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Methodology

**Preprocessing**: Duplicates are removed, features/target are
separated, and the data is split into train/test sets using a
stratified 80/20 split (`stratify=y`) so both sets preserve the same
fraud ratio. The `Amount` column is scaled with `StandardScaler`,
fit only on the training data to avoid data leakage.

**SMOTE**: Because fraud is only ~0.17% of transactions, models
trained directly on the raw data tend to just predict "genuine" every
time. SMOTE (Synthetic Minority Over-sampling Technique) generates
synthetic fraud examples in the training set only, balancing the
classes so the models can actually learn fraud patterns. The test set
is never resampled, so evaluation reflects real-world conditions.

**Logistic Regression**: A simple, interpretable linear baseline
model.

**Random Forest**: An ensemble of decision trees that usually
captures non-linear patterns better than a single linear model, at
the cost of interpretability.

**Isolation Forest**: An unsupervised anomaly-detection model that
isolates outliers (potential fraud) without needing labels during
training, useful as a comparison point to the supervised models.

**Evaluation metrics**: Precision, Recall, F1-score, and PR-AUC are
used instead of accuracy, because accuracy is misleading on a dataset
this imbalanced (predicting "genuine" for every transaction would
already give ~99.8% accuracy while catching zero fraud).

## Deployment

This application can be deployed for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this project to a GitHub repository (include the `models/`
   folder with the trained `.pkl` files).
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Click "New app", select the repository, branch, and set the main
   file path to `app.py`.
4. Deploy. Streamlit Cloud will install everything listed in
   `requirements.txt` automatically.

All file paths in this project are relative (e.g. `data/creditcard.csv`,
`models/fraud_model.pkl`), so the app works the same way locally and
on Streamlit Cloud.

## Likely viva questions

- **Why not accuracy?** Predicting "genuine" every time already gives ~99.8%.
- **Why SMOTE only on training data?** Resampling the test set would make results look better than real life.
- **Why fit the scaler on training data only?** To avoid data leakage from the test set.
- **Why PR-AUC over ROC-AUC?** With so few positives, ROC-AUC can look great even when precision is poor.
- **What does the threshold slider do?** Lower catches more fraud (higher recall) but raises false alarms (lower precision).
- **Why is Isolation Forest not deployed?** It is unsupervised and ignores the labels, so it scores lower; it is a comparison point.

## Fonts

Hanken Grotesk and Source Serif are bundled in `static/fonts/` and loaded
through `.streamlit/config.toml`, so they work offline and on Streamlit
Cloud. Both are licensed under the SIL Open Font License.
