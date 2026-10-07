"""
app.py
------
Streamlit web application for the Financial Fraud Detection project
("Fraud Desk").

Run with:
    streamlit run app.py

The app only needs the files produced by `python train.py`
(models/*.pkl, models/*.json, models/sample_transactions.csv),
so it also works on Streamlit Cloud where the big dataset is not uploaded.

Theme and fonts are configured in .streamlit/config.toml; the fonts
themselves are in static/fonts/.
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "fraud_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")
EDA_PATH = os.path.join("models", "eda_summary.json")
SAMPLES_PATH = os.path.join("models", "sample_transactions.csv")
FONT_DIR = os.path.join("static", "fonts")

V_FEATURES = [f"V{i}" for i in range(1, 29)]

# Colour palette: warm paper, ink blue, and "stamp" red for alerts
PAPER = "#F6F3EC"
CARD = "#FFFDF8"
INK = "#16213A"
LINE = "#DDD6C6"
ALERT = "#C8402F"
SAFE = "#2F7D6B"
WARN = "#D9962B"
SLATE = "#5B6B8C"
MUTED = "#6F7480"

st.set_page_config(
    page_title="Fraud Desk | Transaction Review",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling: CSS for the custom components + matplotlib theme
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
  --paper:#F6F3EC; --card:#FFFDF8; --ink:#16213A; --line:#DDD6C6;
  --alert:#C8402F; --safe:#2F7D6B; --warn:#D9962B; --muted:#6F7480;
  --serif:'Source Serif', Georgia, serif;
  --mono:ui-monospace, 'SF Mono', Consolas, 'Courier New', monospace;
}
.block-container { padding-top: 2.4rem; max-width: 1100px; }
#MainMenu, footer, .stAppDeployButton { visibility: hidden; display: none; }

/* page header: small mono label, serif title, thin rule */
.eyebrow { font-family: var(--mono); font-size: .74rem; letter-spacing: .16em;
           text-transform: uppercase; color: var(--alert); margin-bottom: .35rem; }
.page-title { font-family: var(--serif); font-weight: 700; font-size: 2.45rem;
              line-height: 1.12; color: var(--ink); margin: 0 0 .5rem 0; }
.page-sub { color: var(--muted); font-size: 1.08rem; max-width: 44rem; margin: 0; }
.rule { border: 0; border-top: 3px double var(--ink); margin: 1.1rem 0 1.6rem 0; opacity: .85; }

/* printed-looking cards: small radius, hard offset shadow */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 4px;
        padding: 1.2rem 1.4rem; box-shadow: 4px 4px 0 var(--line); color: var(--ink); }
.card p { margin: 0 0 .6rem 0; line-height: 1.55; }
.card p:last-child { margin-bottom: 0; }
.card h4 { font-family: var(--serif); margin: 0 0 .5rem 0; }

/* home hero */
.hero-title { font-family: var(--serif); font-weight: 700; font-size: 2.3rem;
              line-height: 1.15; color: var(--ink); margin: .2rem 0 .8rem 0; }
.hero-title em { color: var(--alert); font-style: normal; }
.hero-text { font-size: 1.05rem; line-height: 1.6; color: #39435c; }

/* needle in a haystack: 600 dots, one of them is fraud */
.haystack { display: grid; grid-template-columns: repeat(30, 1fr); gap: 4px; }
.haystack i { display: block; aspect-ratio: 1; border-radius: 50%; background: #D3CCBA; }
.haystack i.needle { background: var(--alert); box-shadow: 0 0 0 3px rgba(200,64,47,.28); }
.hay-caption { font-family: var(--mono); font-size: .74rem; color: var(--muted);
               margin-top: .7rem; letter-spacing: .03em; }

/* ledger-style stat cards */
.ledger { background: var(--card); border: 1px solid var(--line);
          border-top: 4px solid var(--ink); border-radius: 3px;
          padding: .8rem 1rem .9rem 1rem; min-height: 98px; }
.ledger.alert { border-top-color: var(--alert); }
.ledger.warn  { border-top-color: var(--warn); }
.ledger.safe  { border-top-color: var(--safe); }
.ledger .label { font-family: var(--mono); font-size: .7rem; letter-spacing: .12em;
                 text-transform: uppercase; color: var(--muted); }
.ledger .value { font-family: var(--serif); font-weight: 700; font-size: 1.85rem;
                 color: var(--ink); line-height: 1.25; }

/* investigation timeline */
.timeline { list-style: none; margin: 0; padding: 0; position: relative; }
.timeline::before { content: ""; position: absolute; left: 17px; top: 8px; bottom: 8px;
                    border-left: 2px dashed #bdb5a2; }
.timeline li { display: flex; gap: 1rem; margin-bottom: 1.1rem; position: relative; }
.timeline .n { flex: 0 0 36px; height: 36px; border-radius: 50%; background: var(--ink);
               color: #fff; font-family: var(--serif); font-weight: 700;
               display: flex; align-items: center; justify-content: center; z-index: 1; }
.timeline b { font-family: var(--serif); font-size: 1.08rem; color: var(--ink); }
.timeline span { color: #4a5470; display: block; line-height: 1.5; }

/* the transaction ticket (the form) */
div[data-testid="stForm"] { background: var(--card); border: 2px dashed #b3ab98;
                            border-radius: 4px; padding: 1.2rem 1.4rem; }
.ticket-head { font-family: var(--mono); font-size: .74rem; letter-spacing: .18em;
               text-transform: uppercase; color: var(--muted);
               border-bottom: 1px dashed #b3ab98; padding-bottom: .5rem; margin-bottom: .8rem; }

/* verdict: stamp + risk meter */
.verdict { display: flex; gap: 1.6rem; align-items: center; background: var(--card);
           border: 1px solid var(--line); border-radius: 4px; padding: 1.4rem 1.6rem;
           box-shadow: 4px 4px 0 var(--line); margin-top: .8rem; color: var(--ink); }
.stamp { flex: 0 0 auto; font-family: var(--mono); font-weight: 800; font-size: 1.35rem;
         letter-spacing: .22em; padding: .45rem 1rem .45rem 1.2rem; border: 4px solid;
         border-radius: 6px; transform: rotate(-6deg); opacity: .92; }
.stamp.bad  { color: var(--alert); border-color: var(--alert); }
.stamp.good { color: var(--safe);  border-color: var(--safe); }
.verdict h3 { font-family: var(--serif); margin: 0 0 .25rem 0; font-size: 1.5rem; }
.verdict p  { margin: 0 0 .9rem 0; color: #39435c; }
.meter { position: relative; display: flex; height: 12px; border-radius: 2px; overflow: visible; }
.meter .z1 { width: 30%; background: var(--safe); }
.meter .z2 { width: 40%; background: var(--warn); }
.meter .z3 { width: 30%; background: var(--alert); }
.meter .pin { position: absolute; top: -9px; width: 0; height: 0;
              border-left: 7px solid transparent; border-right: 7px solid transparent;
              border-top: 11px solid var(--ink); transform: translateX(-7px); }
.meter-scale { display: flex; justify-content: space-between; font-family: var(--mono);
               font-size: .68rem; color: var(--muted); margin-top: .35rem; }
.verdict-main { flex: 1 1 auto; }

/* sidebar */
.brand { display: flex; gap: .7rem; align-items: center; margin: .2rem 0 1.4rem 0; }
.brand-name { font-family: var(--serif); font-weight: 700; font-size: 1.35rem;
              color: #F3EEE2; line-height: 1.1; }
.brand-sub { font-family: var(--mono); font-size: .64rem; letter-spacing: .14em;
             text-transform: uppercase; color: #9AA3BA; margin-top: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: .5rem .75rem; border-radius: 4px; border-left: 3px solid transparent;
    margin-bottom: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.06); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,.10); border-left-color: var(--alert); }
.model-chip { font-family: var(--mono); font-size: .66rem; letter-spacing: .14em;
              text-transform: uppercase; color: #9AA3BA; margin-top: 1.6rem; }
.model-name { font-family: var(--serif); font-size: 1.1rem; color: #F3EEE2; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def setup_chart_style():
    """Register the bundled font with matplotlib and apply the paper theme."""
    for fname in ("HankenGrotesk-Regular.otf", "HankenGrotesk-SemiBold.otf"):
        path = os.path.join(FONT_DIR, fname)
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": ["Hanken Grotesk", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": LINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


setup_chart_style()


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def page_header(eyebrow, title, subtitle=""):
    sub = f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1 class="page-title">{title}</h1>{sub}<hr class="rule">',
        unsafe_allow_html=True,
    )


def ledger(label, value, style=""):
    return (f'<div class="ledger {style}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


def show_fig(fig):
    """Render a matplotlib figure and free its memory."""
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_samples():
    return pd.read_csv(SAMPLES_PATH) if os.path.exists(SAMPLES_PATH) else None


def classify_risk(prob):
    """Map a fraud probability to a simple risk band."""
    if prob < 0.30:
        return "LOW", SAFE
    elif prob < 0.70:
        return "MEDIUM", WARN
    return "HIGH", ALERT


def predict(row_df, model, scaler, feature_order, threshold):
    """
    Same preprocessing as training: keep the training column order,
    scale 'Amount' with the saved scaler, then use the model's
    fraud probability and the chosen decision threshold.
    """
    df = row_df[feature_order].copy()
    df["Amount"] = scaler.transform(df[["Amount"]])
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs


model = load_model()
scaler = load_scaler()
metrics = load_json(METRICS_PATH)
eda = load_json(EDA_PATH)
samples = load_samples()
feature_order = metrics["feature_order"] if metrics else ["Time"] + V_FEATURES + ["Amount"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
BRAND = """
<div class="brand">
  <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#F3EEE2"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 2.5 20.5 5.6v6c0 4.7-3.5 8.3-8.5 9.9-5-1.6-8.5-5.2-8.5-9.9v-6z"/>
    <circle cx="11.2" cy="11" r="3.2" stroke="#C8402F"/>
    <path d="m13.6 13.4 2.6 2.6" stroke="#C8402F"/>
  </svg>
  <div><div class="brand-name">Fraud Desk</div>
  <div class="brand-sub">Transaction review</div></div>
</div>
"""
st.sidebar.markdown(BRAND, unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Check a Transaction", "Analytics", "Model Performance", "About"],
    label_visibility="collapsed",
)
if metrics:
    st.sidebar.markdown(
        f'<div class="model-chip">Model on duty</div>'
        f'<div class="model-name">{metrics["best_model"]}</div>',
        unsafe_allow_html=True,
    )


def need_training_warning():
    st.warning(
        "Trained files were not found. Run `python train.py` once "
        "(with `data/creditcard.csv` in place) and refresh this page."
    )


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    page_header("Project overview", "Financial Fraud Detection")

    if eda is None:
        need_training_warning()
        n_ratio, total, fraud, genuine = 600, 0, 0, 0
    else:
        total, fraud, genuine = eda["total"], eda["fraud"], eda["genuine"]
        n_ratio = max(1, round(total / max(fraud, 1)))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="card" style="height:100%">
              <div class="eyebrow">The problem</div>
              <div class="hero-title">1 in {n_ratio:,} card transactions is fraud.
                <em>Finding it is the whole job.</em></div>
              <p class="hero-text">Fraud is rare, but each missed case costs real money, and
              nobody can read a million transactions by hand. This project learns what
              fraud looks like from {total:,} real transactions, then reviews new ones the
              way a careful analyst would: it gives a probability, a risk level and a
              clear flag.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        needle = 387
        dots = "".join('<i class="needle"></i>' if i == needle else "<i></i>" for i in range(600))
        st.markdown(
            f"""
            <div class="card">
              <div class="haystack">{dots}</div>
              <div class="hay-caption">EACH DOT = ONE TRANSACTION. ONE OF THEM IS FRAUD.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if eda is not None:
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(ledger("Transactions", f"{total:,}"), unsafe_allow_html=True)
        c2.markdown(ledger("Genuine", f"{genuine:,}", "safe"), unsafe_allow_html=True)
        c3.markdown(ledger("Fraud cases", f"{fraud:,}", "alert"), unsafe_allow_html=True)
        c4.markdown(ledger("Fraud rate", f"{fraud / total * 100:.3f}%", "warn"), unsafe_allow_html=True)

    st.markdown("")
    st.subheader("How the investigation works")
    st.markdown(
        """
        <ol class="timeline">
          <li><div class="n">1</div><div><b>Clean the records</b>
              <span>Drop duplicate rows and make sure nothing is missing.</span></div></li>
          <li><div class="n">2</div><div><b>Split and scale</b>
              <span>Hold back 20% of transactions for testing, never shown to the model.
              Put <i>Amount</i> on a common scale using the training data only.</span></div></li>
          <li><div class="n">3</div><div><b>Even out the odds</b>
              <span>SMOTE adds synthetic fraud examples to the training set so the model
              has enough fraud to learn from.</span></div></li>
          <li><div class="n">4</div><div><b>Let three models compete</b>
              <span>Logistic Regression, Random Forest and Isolation Forest are scored on
              the same untouched test set.</span></div></li>
          <li><div class="n">5</div><div><b>Put the winner on duty</b>
              <span>The best model is saved and reviews transactions on the next page.</span></div></li>
        </ol>
        """,
        unsafe_allow_html=True,
    )
    st.info("Ready to try it? Open **Check a Transaction** in the sidebar and press "
            "*Fraud example*.")

# ---------------------------------------------------------------------------
# CHECK A TRANSACTION
# ---------------------------------------------------------------------------
elif page == "Check a Transaction":
    page_header("Case desk", "Check a Transaction",
                "Fill in a transaction, or load a real example, and see how the model reads it.")

    if model is None or scaler is None:
        need_training_warning()
        st.stop()

    threshold = st.slider(
        "How suspicious before we flag it?",
        0.05, 0.95, 0.50, 0.05, format="%.2f",
        help="A transaction is flagged when its fraud probability is at or above this value. "
             "Lower catches more fraud but raises more false alarms.",
    )

    tab1, tab2 = st.tabs(["Single transaction", "Upload a CSV"])

    # -- single transaction -----------------------------------------------
    with tab1:
        # Default value for every input box (set once, before the widgets exist)
        for c in feature_order:
            st.session_state.setdefault(f"in_{c}", 0.0)

        if samples is not None:
            st.caption("Load a real transaction from the test set. These examples are ones the "
                       "model classifies correctly, so the demo is easy to follow. No model "
                       "catches every fraud; the Model Performance page shows the real numbers.")
            b1, b2, b3, _ = st.columns([1, 1, 1, 2])
            pick = None
            if b1.button("Genuine example", width="stretch"):
                pick = samples[samples["Class"] == 0].sample(1).iloc[0]
            if b2.button("Fraud example", width="stretch"):
                pick = samples[samples["Class"] == 1].sample(1).iloc[0]
            if b3.button("Clear", width="stretch"):
                pick = pd.Series({c: 0.0 for c in feature_order})
            if pick is not None:  # must be set BEFORE the widgets are drawn
                for c in feature_order:
                    st.session_state[f"in_{c}"] = float(pick[c])

        with st.form("single_form"):
            st.markdown('<div class="ticket-head">Transaction ticket</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            time_val = c1.number_input("Time (seconds since the first transaction)",
                                       key="in_Time", step=1.0)
            amount_val = c2.number_input("Amount", key="in_Amount", step=1.0, min_value=0.0)

            with st.expander("Anonymised features V1 – V28"):
                st.caption("These are PCA components. The bank hides what they mean to protect "
                           "customers, so use an example above or leave them at 0.")
                cols = st.columns(4)
                v_values = {}
                for i, feat in enumerate(V_FEATURES):
                    v_values[feat] = cols[i % 4].number_input(
                        feat, key=f"in_{feat}", step=0.1, format="%.3f")

            submitted = st.form_submit_button("Review transaction", type="primary")

        if submitted:
            row = pd.DataFrame([{"Time": time_val, **v_values, "Amount": amount_val}])
            try:
                preds, probs = predict(row, model, scaler, feature_order, threshold)
                prob, pred = float(probs[0]), int(preds[0])
                level, colour = classify_risk(prob)

                if pred == 1:
                    stamp, css = "FLAGGED", "bad"
                    title = "Likely fraudulent"
                    note = (f"The model puts the fraud probability at {prob * 100:.1f}%, above your "
                            f"{threshold * 100:.0f}% flag line. Worth a human review.")
                else:
                    stamp, css = "CLEARED", "good"
                    title = "Looks genuine"
                    note = (f"The model puts the fraud probability at {prob * 100:.1f}%, below your "
                            f"{threshold * 100:.0f}% flag line. Nothing unusual found.")

                st.markdown(
                    f"""
                    <div class="verdict">
                      <div class="stamp {css}">{stamp}</div>
                      <div class="verdict-main">
                        <h3>{title} <span style="color:{colour};font-size:.95rem;
                            font-family:var(--mono);letter-spacing:.1em">· {level} RISK</span></h3>
                        <p>{note}</p>
                        <div class="meter">
                          <div class="z1"></div><div class="z2"></div><div class="z3"></div>
                          <div class="pin" style="left:{min(max(prob, 0.005), 0.995) * 100:.1f}%"></div>
                        </div>
                        <div class="meter-scale"><span>0%</span><span>30%</span>
                          <span>70%</span><span>100%</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("A model estimate, not proof of fraud.")
            except Exception as e:
                st.error(f"Could not generate a prediction: {e}")

    # -- batch CSV -----------------------------------------------------------
    with tab2:
        st.write("Upload a CSV with the columns `Time, V1 … V28, Amount`. "
                 "An optional `Class` column is ignored.")
        if samples is not None:
            st.download_button(
                "Download a sample CSV to try",
                data=samples.drop(columns=["Class"]).to_csv(index=False).encode("utf-8"),
                file_name="sample_transactions.csv",
                mime="text/csv",
            )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read that file: {e}")
                st.stop()

            missing = [c for c in feature_order if c not in batch_df.columns]
            if missing:
                st.error("The file is missing these columns: " + ", ".join(missing))
            elif batch_df.empty:
                st.warning("The file has no rows.")
            elif batch_df[feature_order].isnull().any().any():
                st.error("The file contains empty cells. Please fill or remove them.")
            else:
                try:
                    preds, probs = predict(batch_df, model, scaler, feature_order, threshold)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                result = batch_df.copy()
                result["Fraud Probability"] = probs.round(4)
                result["Prediction"] = np.where(preds == 1, "Fraud", "Genuine")
                result["Risk Level"] = [classify_risk(p)[0] for p in probs]

                n_fraud = int((preds == 1).sum())
                c1, c2, c3 = st.columns(3)
                c1.markdown(ledger("Rows reviewed", f"{len(result):,}"), unsafe_allow_html=True)
                c2.markdown(ledger("Flagged as fraud", f"{n_fraud:,}", "alert"), unsafe_allow_html=True)
                c3.markdown(ledger("Flagged share", f"{n_fraud / len(result) * 100:.1f}%", "warn"),
                            unsafe_allow_html=True)

                st.markdown("")
                show_flagged = st.checkbox("Show only flagged transactions")
                shown = result[result["Prediction"] == "Fraud"] if show_flagged else result
                front = ["Prediction", "Risk Level", "Fraud Probability"]
                st.dataframe(shown[front + [c for c in shown.columns if c not in front]],
                             width="stretch", hide_index=True)

                st.download_button(
                    "Download full results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------
elif page == "Analytics":
    page_header("Evidence", "What the data shows",
                "A look at the transactions before any model sees them.")

    if eda is None:
        need_training_warning()
        st.stop()

    n_ratio = max(1, round(eda["total"] / max(eda["fraud"], 1)))
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(ledger("Transactions", f"{eda['total']:,}"), unsafe_allow_html=True)
    c2.markdown(ledger("Fraud cases", f"{eda['fraud']:,}", "alert"), unsafe_allow_html=True)
    c3.markdown(ledger("Fraud rate", f"{eda['fraud'] / eda['total'] * 100:.3f}%", "warn"),
                unsafe_allow_html=True)
    c4.markdown(ledger("Duplicates removed", f"{eda['duplicates_removed']:,}"), unsafe_allow_html=True)
    st.markdown("")

    def draw_hist(ax, h, colour, label=None, density=False):
        counts = np.array(h["counts"], dtype=float)
        edges = np.array(h["edges"])
        if density and counts.sum() > 0:
            counts = counts / counts.sum() * 100
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=colour, alpha=0.9, label=label, edgecolor=PAPER, linewidth=0.5)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Fraud is a needle in a haystack")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        vals = [eda["genuine"], eda["fraud"]]
        bars = ax.bar(["Genuine", "Fraud"], vals, color=[SLATE, ALERT], width=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("Transactions (log scale)")
        ax.grid(axis="x", visible=False)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:,}", ha="center", va="bottom",
                    fontsize=10, fontweight="semibold")
        ax.set_ylim(top=max(vals) * 12)
        ax.annotate(f"about 1 in {n_ratio:,}", xy=(1, vals[1]), xytext=(0.45, vals[1] * 6),
                    arrowprops=dict(arrowstyle="->", color=INK), fontsize=9.5, color=INK)
        show_fig(fig)
        st.caption("A log scale is the only way to see both bars. This imbalance is why we use "
                   "SMOTE and why accuracy is a misleading score here.")

    with right:
        st.subheader("Most purchases are small")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        draw_hist(ax, eda["amount_hist_all"], SLATE)
        ax.set_xlabel("Amount (up to the 99th percentile)")
        ax.set_ylabel("Transactions")
        ax.grid(axis="x", visible=False)
        show_fig(fig)
        st.caption("A long tail of large amounts is clipped so the bulk of the data stays readable.")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("What do fraud amounts look like?")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        draw_hist(ax, eda["amount_hist_fraud"], ALERT)
        ax.set_xlabel("Amount")
        ax.set_ylabel("Fraud transactions")
        ax.grid(axis="x", visible=False)
        s = eda["amount_stats"]
        ax.text(0.97, 0.94, f"median fraud: {s['fraud_median']:,.2f}\nmedian genuine: {s['genuine_median']:,.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="square,pad=0.5", fc=CARD, ec=LINE))
        show_fig(fig)
        st.caption("Fraud is not simply big money, which is why amount alone cannot catch it.")

    with right:
        st.subheader("When does it happen?")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        draw_hist(ax, eda["time_hist_genuine"], SLATE, "Genuine", density=True)
        draw_hist(ax, eda["time_hist_fraud"], ALERT, "Fraud", density=True)
        ax.set_xlabel("Hours since the first transaction")
        ax.set_ylabel("% of that class")
        ax.grid(axis="x", visible=False)
        ax.legend()
        show_fig(fig)
        st.caption("Each class is a percentage of its own total, so the two can be compared fairly.")

    st.subheader("Which features move with fraud?")
    corr = eda["top_correlations"]
    names = list(corr.keys())[::-1]
    vals = [corr[n] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(names, vals, color=[ALERT if v > 0 else SLATE for v in vals], height=0.65)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Correlation with Class (red: higher value, more fraud; blue: lower value, more fraud)")
    ax.grid(axis="y", visible=False)
    show_fig(fig)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    page_header("Scorecard", "How well does it catch fraud?")

    if metrics is None:
        need_training_warning()
        st.stop()

    best = metrics["best_model"]
    res = metrics["results"]
    b = res[best]
    test_size = metrics.get("test_size")
    extra = f" It was scored on {test_size:,} transactions it had never seen." if test_size else ""

    st.markdown(
        f'<div class="card"><p><b>{best}</b> is the model on duty: it had the highest F1-score '
        f'of the supervised models.{extra}</p>'
        f'<p>In plain English: out of every 100 real frauds it catches about '
        f'<b>{b["recall"] * 100:.0f}</b>, and when it raises an alarm it is right about '
        f'<b>{b["precision"] * 100:.0f}</b> times out of 100.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(ledger("Precision", f"{b['precision']:.3f}"), unsafe_allow_html=True)
    c2.markdown(ledger("Recall", f"{b['recall']:.3f}", "safe"), unsafe_allow_html=True)
    c3.markdown(ledger("F1-score", f"{b['f1_score']:.3f}", "warn"), unsafe_allow_html=True)
    c4.markdown(ledger("PR-AUC", f"{b['pr_auc']:.3f}", "alert"), unsafe_allow_html=True)
    st.caption("Precision: of the transactions we flag, how many are really fraud. "
               "Recall: of all real fraud, how much we catch.")
    st.markdown("")

    colours = {"Logistic Regression": SLATE, "Random Forest": ALERT, "Isolation Forest": WARN}
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Confusion matrix")
        cm = np.array(b["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("ink", [CARD, SLATE]))
        ax.grid(False)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Genuine", "Fraud"]); ax.set_yticklabels(["Genuine", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                dark = cm[i, j] > cm.max() / 2
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                        fontweight="semibold", color="white" if dark else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        show_fig(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{tp:,} frauds caught, {fn:,} missed, {fp:,} false alarms.")

    with right:
        st.subheader("Precision–recall curves")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        for name, r in res.items():
            ax.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"], lw=2.2,
                    color=colours.get(name, MUTED), label=f"{name} ({r['pr_auc']:.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.03)
        ax.legend(fontsize=8, loc="lower left")
        show_fig(fig)
        st.caption("The closer a line hugs the top-right corner, the better. PR-AUC is in brackets.")

    st.subheader("All three models, side by side")
    names = list(res.keys())
    metric_keys = [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1"), ("pr_auc", "PR-AUC")]
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    width = 0.2
    x = np.arange(len(names))
    palette = [SLATE, SAFE, WARN, ALERT]
    for k, ((key, label), colour) in enumerate(zip(metric_keys, palette)):
        ax.bar(x + (k - 1.5) * width, [res[n][key] for n in names], width, label=label, color=colour)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.18)
    ax.legend(ncol=4, loc="upper right")
    ax.grid(axis="x", visible=False)
    show_fig(fig)

    table = pd.DataFrame([
        {"Model": n, "Precision": r["precision"], "Recall": r["recall"],
         "F1 Score": r["f1_score"], "PR-AUC": r["pr_auc"], "ROC-AUC": r.get("roc_auc")}
        for n, r in res.items()
    ]).round(4)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Isolation Forest is unsupervised (it never sees the labels), so it is shown for "
               "comparison only and is not put on duty.")

    fi = metrics.get("feature_importance")
    if fi:
        st.subheader(f"What the {best} pays attention to")
        feats = fi["features"][::-1]
        vals = fi["values"][::-1]
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        ax.barh(feats, vals, color=INK, height=0.65)
        ax.set_xlabel(fi["label"])
        ax.grid(axis="y", visible=False)
        show_fig(fig)

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    page_header("Case notes", "About this project",
                "Python for Data Science · B.Tech Semester 5 · PBL")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="card">
              <h4>The problem</h4>
              <p>Fraudulent card transactions cause large financial losses, and checking millions
              of them by hand is impossible. A model can screen them automatically and hand only
              the suspicious ones to a person.</p>
              <h4 style="margin-top:1rem">The dataset</h4>
              <p>Kaggle <i>Credit Card Fraud Detection</i> (mlg-ulb): transactions by European
              cardholders over two days, with only about 0.17% fraud.</p>
              <p><code>Time</code> seconds since the first transaction<br>
              <code>V1–V28</code> PCA features (meaning hidden)<br>
              <code>Amount</code> transaction amount<br>
              <code>Class</code> 0 = genuine, 1 = fraud</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h4>Decisions worth defending</h4>
              <p><b>Stratified split</b> keeps the same fraud ratio in train and test.</p>
              <p><b>Scaler fitted on training data only</b>, so nothing leaks from the test set.</p>
              <p><b>SMOTE on the training set only</b>; the test set stays realistic.</p>
              <p><b>Precision, recall, F1 and PR-AUC</b> instead of accuracy, because always
              guessing &ldquo;genuine&rdquo; is already about 99.8% accurate.</p>
              <p><b>Adjustable threshold</b> lets you trade missed fraud against false alarms.</p>
              <h4 style="margin-top:1rem">Limitations</h4>
              <p>The features are anonymised, the data is from 2013, and one trained model cannot
              adapt to new fraud patterns without retraining. A real system keeps a human in
              the loop.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.caption("Built with Python, Pandas, NumPy, Matplotlib, Scikit-learn, imbalanced-learn, "
               "Joblib and Streamlit. Fonts: Hanken Grotesk and Source Serif (SIL Open Font "
               "License). A student project: it never handles real card numbers, CVVs, PINs or "
               "banking credentials.")
