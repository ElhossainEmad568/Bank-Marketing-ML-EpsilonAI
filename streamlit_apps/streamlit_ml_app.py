import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Bank Subscription Predictor", layout="wide")

# Shared color theme for charts
PRIMARY_COLOR = "#1F6FEB"
COLOR_SEQUENCE = ["#1F6FEB", "#2EC4B6", "#F4A259", "#9B5DE5", "#E63946", "#0B2545"]
PREDICTION_COLORS = {"yes": "#2EC4B6", "no": "#0B2545"}

# Section styling: hero banner, KPI cards, section headers
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg, #0B2545, #1F6FEB);
        padding: 24px 28px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0; }
    .hero p { margin: 6px 0 0 0; opacity: 0.9; }
    .kpi-card {
        background: #1F6FEB;
        color: white;
        padding: 18px;
        border-radius: 8px;
        text-align: center;
    }
    .kpi-card .value { font-size: 26px; font-weight: 700; }
    .kpi-card .label { font-size: 13px; opacity: 0.9; }
    .section-header {
        background: #0B2545;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        margin: 24px 0 12px 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "trained_models/best_model_without_duration.joblib"
FEATURES_PATH = "saved_transformers/feature_list_no_duration.joblib"
THRESHOLD_PATH = "saved_transformers/final_threshold.pkl"


@st.cache_resource
def load_artifacts():
    # `model` is a full Pipeline (preprocessor + estimator), so raw input
    # DataFrames can be passed directly to predict / predict_proba.
    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)
    try:
        threshold = joblib.load(THRESHOLD_PATH)
    except FileNotFoundError:
        threshold = 0.5
    return model, feature_list, threshold


@st.cache_data
def load_data():
    cleaned_path = "cleaned_bank_dataset.csv"
    raw_path = "bank/bank-full.csv"
    try:
        df = pd.read_csv(cleaned_path)
    except FileNotFoundError:
        df = pd.read_csv(raw_path, sep=";")
    return df


@st.cache_data
def score_dataset(df, _model, feature_list, threshold):
    scored = df.copy()
    scored["predicted_prob"] = _model.predict_proba(df.reindex(columns=feature_list))[:, 1]
    scored["predicted_label"] = (scored["predicted_prob"] >= threshold).map({True: "yes", False: "no"})
    return scored


try:
    model, feature_list, threshold = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run the ML notebook to generate them.")
    st.stop()

df = load_data()
scored_df = score_dataset(df, model, feature_list, threshold)

# ---- Sidebar: filters ----
st.sidebar.markdown("## \U0001F4CB Prediction Filters")
st.sidebar.divider()

st.sidebar.markdown("**Demographics**")
job = st.sidebar.multiselect("Job", sorted(scored_df["job"].dropna().unique()))
marital = st.sidebar.multiselect("Marital", sorted(scored_df["marital"].dropna().unique()))
education = st.sidebar.multiselect("Education", sorted(scored_df["education"].dropna().unique()))

st.sidebar.markdown("**Financial**")
housing = st.sidebar.multiselect("Housing Loan", sorted(scored_df["housing"].dropna().unique()))
loan = st.sidebar.multiselect("Personal Loan", sorted(scored_df["loan"].dropna().unique()))

st.sidebar.markdown("**Campaign**")
contact = st.sidebar.multiselect("Contact Type", sorted(scored_df["contact"].dropna().unique()))
month = st.sidebar.multiselect("Month", sorted(scored_df["month"].dropna().unique()))

age_min, age_max = int(scored_df["age"].min()), int(scored_df["age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

filters = {
    "job": job, "marital": marital, "education": education,
    "housing": housing, "loan": loan, "contact": contact, "month": month,
}
filtered_df = scored_df.copy()
for col, selected in filters.items():
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]
filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])]

st.sidebar.divider()
st.sidebar.write("**Model:** Tuned XGBoost (no-duration)")
st.sidebar.write(f"**Decision threshold:** {threshold:.2f}")
st.sidebar.caption("Epsilon AI ML Program")

# ---- Hero banner ----
st.markdown(
    f"""
    <div class="hero">
        <h1>\U0001F3E6 Term Deposit Subscription Predictor</h1>
        <p>Pre-Campaign Prediction Explorer &middot; {len(scored_df):,} Customers &middot; Model: Tuned XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- KPI row ----
if len(filtered_df) > 0:
    actual_rate = (filtered_df["y"] == "yes").mean()
    predicted_rate = (filtered_df["predicted_label"] == "yes").mean()
    avg_prob = filtered_df["predicted_prob"].mean()
    high_opportunity_rate = (filtered_df["predicted_prob"] >= 0.6).mean()
else:
    actual_rate = predicted_rate = avg_prob = high_opportunity_rate = 0.0

kpis = [
    (f"{len(filtered_df):,}", "Customers"),
    (f"{actual_rate:.1%}", "Actual Subscription Rate"),
    (f"{predicted_rate:.1%}", "Predicted Subscription Rate"),
    (f"{avg_prob:.1%}", "Avg Predicted Probability"),
    (f"{high_opportunity_rate:.1%}", "High Opportunity (>=60%)"),
    (f"{threshold:.2f}", "Decision Threshold"),
]
for col, (value, label) in zip(st.columns(6), kpis):
    col.markdown(
        f"<div class='kpi-card'><div class='value'>{value}</div><div class='label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

# ---- Probability distribution ----
st.markdown("<div class='section-header'>\U0001F4CA Predicted Probability Distribution</div>", unsafe_allow_html=True)
if len(filtered_df) > 0:
    fig_prob = px.histogram(
        filtered_df, x="predicted_prob", color="predicted_label", nbins=40,
        title="Predicted Subscription Probability",
        labels={"predicted_prob": "Predicted Probability", "predicted_label": "Predicted"},
        color_discrete_map=PREDICTION_COLORS,
    )
    fig_prob.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
    st.plotly_chart(fig_prob, use_container_width=True)
else:
    st.info("No customers match the selected filters.")

# ---- Predicted subscription rate by segment ----
st.markdown("<div class='section-header'>\U0001F3AF Predicted Subscription Rate by Segment</div>", unsafe_allow_html=True)
if len(filtered_df) > 0:
    s1, s2, s3 = st.columns(3)
    with s1:
        job_rate = filtered_df.groupby("job")["predicted_prob"].mean().reset_index(name="avg_predicted_prob")
        fig_job = px.bar(
            job_rate, x="job", y="avg_predicted_prob", title="Avg Predicted Probability by Job",
            text_auto=".1%", color="job", color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig_job.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_job, use_container_width=True)
    with s2:
        edu_rate = filtered_df.groupby("education")["predicted_prob"].mean().reset_index(name="avg_predicted_prob")
        fig_edu = px.bar(
            edu_rate, x="education", y="avg_predicted_prob", title="Avg Predicted Probability by Education",
            text_auto=".1%", color="education", color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig_edu.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)
    with s3:
        contact_rate = filtered_df.groupby("contact")["predicted_prob"].mean().reset_index(name="avg_predicted_prob")
        fig_contact = px.bar(
            contact_rate, x="contact", y="avg_predicted_prob", title="Avg Predicted Probability by Contact Type",
            text_auto=".1%", color="contact", color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig_contact.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_contact, use_container_width=True)
else:
    st.info("No customers match the selected filters.")

# ---- Top opportunities ----
st.markdown("<div class='section-header'>\U0001F31F Top Opportunity Customers</div>", unsafe_allow_html=True)
if len(filtered_df) > 0:
    top_customers = filtered_df.sort_values("predicted_prob", ascending=False).head(10)
    st.dataframe(
        top_customers[["age", "job", "marital", "education", "balance", "contact", "month", "predicted_prob", "predicted_label"]]
        .rename(columns={"predicted_prob": "predicted_probability", "predicted_label": "predicted_subscription"})
        .reset_index(drop=True)
    )
else:
    st.info("No customers match the selected filters.")

# ---- Model performance on filtered data ----
st.markdown("<div class='section-header'>\U00002705 Model Performance on Filtered Data</div>", unsafe_allow_html=True)
if len(filtered_df) > 0:
    y_true = (filtered_df["y"] == "yes").astype(int)
    y_pred = (filtered_df["predicted_label"] == "yes").astype(int)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
    p2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
    p3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
    p4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2%}")
else:
    st.info("No customers match the selected filters.")
