from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"

# Design system: navy, white, and restrained violet accents.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

      :root {
        --ink: #17233b;
        --muted: #65718a;
        --line: #e3e8f2;
        --panel: #ffffff;
        --accent: #6858e8;
        --accent-dark: #5142c5;
      }

      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 95% 2%, rgba(111, 95, 232, 0.09), transparent 22rem),
          #f5f7fc;
        color: var(--ink);
      }
      [data-testid="stHeader"] { background: rgba(245,247,252,0.75); }
      [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
      }
      .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 3.2rem;
      }
      h1, h2, h3, h4 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
      [data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #394761;
        font-size: 0.91rem;
      }
      div[data-baseweb="input"] > div,
      div[data-baseweb="select"] > div,
      div[data-baseweb="number-input"] > div {
        border-radius: 11px;
        border-color: #dce3ef;
        background: #fbfcff;
      }
      div[data-testid="stForm"] {
        border: 1px solid #e1e7f1;
        border-radius: 20px;
        background: rgba(255,255,255,0.94);
        padding: 1.4rem 1.45rem 1.15rem;
        box-shadow: 0 8px 28px rgba(32, 47, 84, 0.035);
      }
      .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem 1.9rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(118deg, #172441 0%, #263765 58%, #584bba 120%);
        box-shadow: 0 18px 40px rgba(30, 43, 81, 0.14);
        margin-bottom: 1.2rem;
      }
      .hero:after {
        content: '';
        position: absolute;
        width: 290px; height: 290px;
        right: -82px; top: -132px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,0.035), 0 0 0 70px rgba(255,255,255,0.025);
        pointer-events: none;
      }
      .hero-kicker {
        display: inline-flex; align-items: center; gap: 9px;
        color: #d2d8ff; letter-spacing: .13em; text-transform: uppercase;
        font-size: .69rem; font-weight: 700; margin-bottom: .8rem;
      }
      .hero-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #a79cff; }
      .hero h1 {
        color: #fff !important;
        font-size: clamp(2rem, 4vw, 3.05rem) !important;
        letter-spacing: -0.045em;
        line-height: 1.08;
        margin: 0 0 .75rem 0;
        max-width: 720px;
      }
      .hero h1 span { color: #bcb4ff; }
      .hero p {
        color: #d9e0f2;
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 650px;
        margin: 0;
      }
      .hero-chips { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.25rem; }
      .hero-chip {
        border: 1px solid rgba(255,255,255,.2);
        background: rgba(255,255,255,.075);
        border-radius: 100px;
        padding: .37rem .7rem;
        font-size: .77rem; color: #f0f2ff;
      }
      .section-eyebrow {
        color: #6c5ce7; font-size: .72rem; letter-spacing: .12em;
        text-transform: uppercase; font-weight: 800; margin: .15rem 0 .35rem;
      }
      .section-title { font-family: 'Manrope',sans-serif; font-size: 1.15rem; font-weight: 800; color: #1c2942; margin: 0 0 .25rem; }
      .section-subtitle { color: #6c7890; font-size: .88rem; margin-bottom: 1.05rem; }
      .stat-card {
        height: 100%; border: 1px solid #e4e9f2; background: white;
        border-radius: 16px; padding: 1rem 1.05rem; min-height: 104px;
        box-shadow: 0 5px 18px rgba(32,47,84,.025);
      }
      .stat-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: #7a859a; font-weight: 800; margin-bottom: .55rem; }
      .stat-value { font-family: 'Manrope',sans-serif; color: #202d49; font-size: 1.2rem; font-weight: 800; letter-spacing: -.025em; }
      .stat-note { color: #8490a4; font-size: .78rem; margin-top: .25rem; }
      .result-card {
        border-radius: 20px; padding: 1.35rem 1.45rem;
        border: 1px solid #dce5f1; background: #fff;
        box-shadow: 0 10px 30px rgba(32,47,84,.045);
        margin-top: .75rem;
      }
      .result-card.exit { border-left: 5px solid #dc5b68; }
      .result-card.stay { border-left: 5px solid #159b78; }
      .result-label { color: #7b879a; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
      .result-headline { color: #1d2a43; font-family: 'Manrope',sans-serif; font-weight: 800; font-size: 1.48rem; margin: .3rem 0 .35rem; }
      .result-copy { color: #65718a; font-size: .92rem; line-height: 1.6; }
      .small-note { color: #7a869a; font-size: .8rem; line-height: 1.55; }
      .side-brand { color: #1e2b47; font-family: 'Manrope',sans-serif; font-size: 1.05rem; font-weight: 800; letter-spacing: -.025em; }
      .side-brand span { color: #6858e8; }
      .side-tag { color: #8a94a7; text-transform: uppercase; letter-spacing: .12em; font-size: .65rem; font-weight: 800; margin-top: .15rem; }
      .side-rule { height: 1px; background: #e7ebf3; margin: 1.1rem 0; }
      .side-heading { color: #26344e; font-weight: 800; font-size: .82rem; margin-bottom: .4rem; }
      .side-copy { color: #6d7890; font-size: .82rem; line-height: 1.6; }
      .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border: 0 !important; border-radius: 12px !important;
        background: linear-gradient(105deg, #6858e8, #5546cb) !important;
        color: white !important; font-weight: 700 !important;
        padding: .72rem 1rem !important;
        box-shadow: 0 7px 15px rgba(94,78,213,.18);
        transition: transform .15s ease, box-shadow .15s ease;
      
      }
      .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 10px 20px rgba(94,78,213,.24); transform: translateY(-1px);
      }
      div[data-testid="stAlert"] { border-radius: 12px; }
      footer { visibility: hidden; }
      @media (max-width: 760px) {
        .hero { padding: 1.45rem; border-radius: 18px; }
        .hero h1 { font-size: 2rem !important; }
        div[data-testid="stForm"] { padding: 1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load trusted model artifacts created by the notebook.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading your trained churn model...")
def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH, FEATURES_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model file(s): " + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    if not isinstance(feature_columns, (list, tuple)) or not all(
        isinstance(col, str) for col in feature_columns
    ):
        raise ValueError("feature_columns.pkl must contain a list of feature names.")

    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and expected_count != len(feature_columns):
        raise ValueError(
            "The saved model and feature_columns.pkl do not match. "
            "Please regenerate all three .pkl files together from the updated notebook."
        )

    scaler_count = getattr(scaler, "n_features_in_", None)
    if scaler_count is not None and scaler_count != len(feature_columns):
        raise ValueError(
            "The saved scaler and feature_columns.pkl do not match. "
            "Please regenerate all three .pkl files together from the updated notebook."
        )

    return model, scaler, list(feature_columns)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">CHURN<span>IQ</span></div>
        <div class="side-tag">Customer analytics demo</div>
        <div class="side-rule"></div>
        <div class="side-heading">About this project</div>
        <div class="side-copy">
          A machine-learning demonstration that estimates whether a bank customer
          may leave, using profile, account, and engagement attributes.
        </div>
        <div class="side-rule"></div>
        <div class="side-heading">Model details</div>
        <div class="side-copy">
          <b>Algorithm:</b> Random Forest<br>
          <b>Estimators:</b> 100 decision trees<br>
          <b>Evaluation split:</b> 80% train / 20% test<br>
          <b>Preprocessing:</b> one-hot encoding + saved scaler
        </div>
        <div class="side-rule"></div>
        <div class="side-heading">How to use</div>
        <div class="side-copy">
          1. Enter the customer details.<br>
          2. Select <b>Run churn assessment</b>.<br>
          3. Review the predicted class and estimated churn probability.
        </div>
        <div class="side-rule"></div>
        <div class="small-note">
          This is an educational prototype. Predictions are estimates from the
          trained model, not guarantees or financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main page hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker"><span class="hero-dot"></span> Machine learning · Customer retention</div>
      <h1>Customer Churn <span>Intelligence</span></h1>
      <p>
        Turn customer and account details into a clear churn prediction.
        Enter a profile below to see whether your Random Forest model predicts
        that the customer is likely to stay or leave.
      </p>
      <div class="hero-chips">
        <span class="hero-chip">Random Forest</span>
        <span class="hero-chip">Interactive prediction</span>
        <span class="hero-chip">Model-based probability</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-eyebrow">Prediction workspace</div>
    <div class="section-title">Customer profile</div>
    <div class="section-subtitle">Complete the fields below. All fields are used only for this prediction.</div>
    """,
    unsafe_allow_html=True,
)

# Load after showing the page header so missing-file errors are easier to understand.
try:
    model, scaler, feature_columns = load_artifacts()
except Exception as exc:
    st.error("The app interface is ready, but the trained model files are not ready yet.")
    st.markdown(
        "Please place these three files in the same folder as `app.py`: "
        "`random_forest_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`. "
        "Create them together by re-running the updated notebook, as explained in `README.md`."
    )
    with st.expander("Technical detail"):
        st.code(str(exc))
    st.stop()

# These are the feature names produced by pd.get_dummies(..., drop_first=True)
# for the Bank Customer Churn dataset used in the supplied notebook.
known_numeric = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
known_encoded = ["Geography_Germany", "Geography_Spain", "Gender_Male"]
required_features = known_numeric + known_encoded
unsupported = [name for name in required_features if name not in feature_columns]
if unsupported:
    st.error(
        "The saved feature list does not match the expected Bank Customer Churn dataset. "
        "Missing feature(s): " + ", ".join(unsupported) + ". Re-run the notebook edits in README.md."
    )
    st.stop()

with st.form("customer_churn_form", clear_on_submit=False, border=True):
    left, middle, right = st.columns(3, gap="large")

    with left:
        st.markdown("<div class='section-eyebrow'>01 / Profile</div>", unsafe_allow_html=True)
        credit_score = st.number_input(
            "Credit score", min_value=300, max_value=900, value=650, step=1,
            help="Use a value in the range supported by the dataset."
        )
        age = st.number_input("Customer age", min_value=18, max_value=100, value=38, step=1)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)

    with middle:
        st.markdown("<div class='section-eyebrow'>02 / Relationship</div>", unsafe_allow_html=True)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
        products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
        has_card = st.selectbox("Has credit card?", ["Yes", "No"], index=0)
        active_member = st.selectbox("Active member?", ["Yes", "No"], index=0)

    with right:
        st.markdown("<div class='section-eyebrow'>03 / Account</div>", unsafe_allow_html=True)
        balance = st.number_input(
            "Account balance", min_value=0.0, max_value=10_000_000.0,
            value=75_000.0, step=1_000.0, format="%.2f",
            help="Enter the value in the same units as the training dataset."
        )
        salary = st.number_input(
            "Estimated salary", min_value=0.0, max_value=10_000_000.0,
            value=60_000.0, step=1_000.0, format="%.2f",
            help="Enter the value in the same units as the training dataset."
        )
        st.markdown(
            "<div class='small-note' style='padding-top:.45rem'>Tip: use realistic values similar to the examples in your training dataset.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run churn assessment  →", use_container_width=True)

if submitted:
    # Build the same dummy-encoded columns used in the notebook. France and Female
    # are the dropped baseline categories because the notebook uses drop_first=True.
    customer = {
        "CreditScore": float(credit_score),
        "Age": float(age),
        "Tenure": float(tenure),
        "Balance": float(balance),
        "NumOfProducts": float(products),
        "HasCrCard": int(has_card == "Yes"),
        "IsActiveMember": int(active_member == "Yes"),
        "EstimatedSalary": float(salary),
        "Geography_Germany": int(geography == "Germany"),
        "Geography_Spain": int(geography == "Spain"),
        "Gender_Male": int(gender == "Male"),
    }

    # This guarantees that the input order matches the model's training order.
    input_df = pd.DataFrame([customer]).reindex(
        columns=feature_columns,
        fill_value=0,
    )

    try:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        churn_probability = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if 1 in classes:
                churn_probability = float(model.predict_proba(scaled_input)[0][classes.index(1)])
    except Exception as exc:
        st.error("The prediction could not be completed. Check that the model, scaler and feature list were saved together.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        st.stop()

    st.markdown("---")
    st.markdown(
        "<div class='section-eyebrow'>Assessment complete</div><div class='section-title'>Prediction results</div>",
        unsafe_allow_html=True,
    )

    if prediction == 1:
        result_class = "exit"
        result_label = "Likely to exit"
        result_copy = "The trained model predicts that this customer may leave the bank. Consider reviewing the relationship and engagement details before making any business decision."
        badge_icon = "↗"
    else:
        result_class = "stay"
        result_label = "Likely to stay"
        result_copy = "The trained model predicts that this customer is more likely to remain with the bank. This is a prediction from historical patterns, not a guarantee."
        badge_icon = "✓"

    result_col, score_col = st.columns([1.35, 1], gap="large")
    with result_col:
        st.markdown(
            f"""
            <div class="result-card {result_class}">
              <div class="result-label">Model decision</div>
              <div class="result-headline">{badge_icon} &nbsp; {result_label}</div>
              <div class="result-copy">{result_copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with score_col:
        if churn_probability is not None:
            probability_width = max(0.0, min(100.0, churn_probability * 100.0))
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-label">Estimated churn probability</div>
                  <div style="font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;color:#202d49;letter-spacing:-.04em;margin:.35rem 0 .8rem">{churn_probability:.1%}</div>
                  <div style="height:9px;background:#edf0f8;border-radius:99px;overflow:hidden;margin-bottom:.85rem">
                    <div style="width:{probability_width:.2f}%;height:100%;background:linear-gradient(90deg,#8c80f4,#6553df);border-radius:99px"></div>
                  </div>
                  <div class="small-note">The model's estimated probability for class 1 (Exited).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("This model does not expose probability scores, so only the predicted class is available.")

    st.markdown(
        "<p class='small-note' style='margin-top:1rem'>Interpretation note: the probability is model-derived and may not be calibrated. It should not be treated as certainty or as a causal explanation for why a customer may leave.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="stat-card" style="margin-top:.8rem">
          <div class="stat-label">Ready when you are</div>
          <div class="stat-value">Enter details to generate a prediction</div>
          <div class="stat-note">No prediction is made until you click the assessment button.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='text-align:center;color:#8a94a7;font-size:.76rem;margin-top:2.2rem'>CHURNIQ · Academic ML project · Random Forest classifier</div>",
    unsafe_allow_html=True,
)
