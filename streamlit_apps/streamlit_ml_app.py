import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Bank Subscription Predictor", layout="wide")

# Shared color theme for charts
PRIMARY_COLOR = "#1F6FEB"
COLOR_SEQUENCE = ["#1F6FEB", "#2EC4B6", "#F4A259", "#9B5DE5", "#E63946", "#0B2545"]
PREDICTION_COLORS = {"yes": "#2EC4B6", "no": "#0B2545"}

# Section styling: hero banner, KPI cards, section headers, result cards
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
    .result-card {
        padding: 24px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 12px;
    }
    .result-card.subscribe { background: linear-gradient(90deg, #2EC4B6, #1F6FEB); }
    .result-card.no-subscribe { background: linear-gradient(90deg, #6c757d, #0B2545); }
    .result-card .headline { font-size: 28px; font-weight: 700; margin: 0; }
    .result-card .subtext { font-size: 15px; opacity: 0.9; margin-top: 6px; }
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


def add_engineered_features(row_df):
    """Recreate the engineered features from raw inputs (see EDA notebook, feature engineering)."""
    out = row_df.copy()
    out["Customer_Engagement_Index"] = out["campaign"] + out["previous"]
    out["Financial_Stability_Index"] = out["balance"] / out["age"]
    out["Marketing_Effectiveness"] = out["duration"] / out["campaign"]
    out["Previous_Campaign_Success"] = (out["poutcome"] == "success").astype(int)
    out["Loan_Risk_Index"] = (out["housing"] == "yes").astype(int) + (out["loan"] == "yes").astype(int)
    return out


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

# ---- Hero banner ----
st.markdown(
    f"""
    <div class="hero">
        <h1>\U0001F3E6 Term Deposit Subscription Predictor</h1>
        <p>Score an individual customer or explore predictions by segment &middot; {len(scored_df):,} Customers &middot; Model: Tuned XGBoost (no duration)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_segments = st.tabs(["\U0001F52E Predict for a Customer", "\U0001F4CA Segment Explorer"])

# ===========================================================================
# Tab 1: Individual prediction via input form
# ===========================================================================
with tab_predict:
    st.markdown("<div class='section-header'>\U0001F4DD Enter Customer Details</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 95, 40)
            job = st.selectbox("Job", sorted(df["job"].dropna().unique()), index=4)
            marital = st.selectbox("Marital Status", sorted(df["marital"].dropna().unique()), index=1)
            education = st.selectbox("Education", sorted(df["education"].dropna().unique()), index=1)

        with c2:
            st.markdown("**Financial**")
            default = st.selectbox("Has Credit in Default?", ["no", "yes"])
            balance = st.number_input("Account Balance (EUR)", min_value=-5000, max_value=20000, value=1000, step=50)
            housing = st.selectbox("Has Housing Loan?", ["no", "yes"], index=1)
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

        with c3:
            st.markdown("**Campaign**")
            contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
            day = st.slider("Day of Month Contacted", 1, 31, 15)
            month = st.selectbox(
                "Month Contacted",
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                index=4,
            )
            campaign = st.number_input("Number of Contacts This Campaign", min_value=1, max_value=50, value=2)

        with c4:
            st.markdown("**Previous Campaign History**")
            previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=275, value=0)
            pdays = st.number_input(
                "Days Since Last Contact (-1 = never contacted)", min_value=-1, max_value=871, value=-1
            )
            poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"])
            duration = st.slider(
                "Expected Call Duration (seconds)", 0, 1200, 235,
                help="Actual call duration is unknown before the call. Use a typical/expected "
                     "value - it is only used to estimate the Marketing_Effectiveness feature.",
            )

        submitted = st.form_submit_button("\U0001F52E Predict Subscription", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": np.nan if contact == "unknown" else contact,
            "day": day, "month": month, "duration": duration, "campaign": campaign,
            "pdays": pdays, "previous": previous,
            "poutcome": np.nan if poutcome == "unknown" else poutcome,
        }])
        input_row = add_engineered_features(input_row)

        prob = float(model.predict_proba(input_row.reindex(columns=feature_list))[:, 1][0])
        will_subscribe = prob >= threshold

        st.markdown("<div class='section-header'>\U0001F3AF Prediction Result</div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1])
        with r1:
            if will_subscribe:
                st.markdown(
                    f"""
                    <div class="result-card subscribe">
                        <p class="headline">✅ Likely to Subscribe</p>
                        <p class="subtext">Predicted probability: {prob:.1%} (>= {threshold:.0%} decision threshold)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("**Recommendation:** Prioritise this customer for outbound contact.")
            else:
                st.markdown(
                    f"""
                    <div class="result-card no-subscribe">
                        <p class="headline">❌ Unlikely to Subscribe</p>
                        <p class="subtext">Predicted probability: {prob:.1%} (< {threshold:.0%} decision threshold)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("**Recommendation:** Lower priority for this campaign; consider re-targeting later.")

        with r2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Predicted Subscription Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PRIMARY_COLOR},
                    "threshold": {
                        "line": {"color": "#E63946", "width": 4},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#E9ECEF"},
                        {"range": [threshold * 100, 100], "color": "#D2F4F0"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<div class='section-header'>\U0001F9EE Engineered Feature Values</div>", unsafe_allow_html=True)
        eng_metrics = [
            ("Customer_Engagement_Index", f"{input_row['Customer_Engagement_Index'].iloc[0]:.2f}", "campaign + previous"),
            ("Financial_Stability_Index", f"{input_row['Financial_Stability_Index'].iloc[0]:.2f}", "balance / age"),
            ("Marketing_Effectiveness", f"{input_row['Marketing_Effectiveness'].iloc[0]:.2f}", "duration / campaign"),
            ("Previous_Campaign_Success", f"{int(input_row['Previous_Campaign_Success'].iloc[0])}", "1 = poutcome success"),
            ("Loan_Risk_Index", f"{int(input_row['Loan_Risk_Index'].iloc[0])}", "housing + personal loan flags"),
        ]
        for col, (name, value, formula) in zip(st.columns(5), eng_metrics):
            col.metric(name.replace("_", " "), value, help=f"Formula: {formula}")
    else:
        st.info("Fill in the customer's details above and click **Predict Subscription** to see the result.")

# ===========================================================================
# Tab 2: Segment explorer with filters
# ===========================================================================
with tab_segments:
    st.sidebar.markdown("## \U0001F4CB Segment Explorer Filters")
    st.sidebar.divider()

    st.sidebar.markdown("**Demographics**")
    f_job = st.sidebar.multiselect("Job", sorted(scored_df["job"].dropna().unique()))
    f_marital = st.sidebar.multiselect("Marital", sorted(scored_df["marital"].dropna().unique()))
    f_education = st.sidebar.multiselect("Education", sorted(scored_df["education"].dropna().unique()))

    st.sidebar.markdown("**Financial**")
    f_housing = st.sidebar.multiselect("Housing Loan", sorted(scored_df["housing"].dropna().unique()))
    f_loan = st.sidebar.multiselect("Personal Loan", sorted(scored_df["loan"].dropna().unique()))

    st.sidebar.markdown("**Campaign**")
    f_contact = st.sidebar.multiselect("Contact Type", sorted(scored_df["contact"].dropna().unique()))
    f_month = st.sidebar.multiselect("Month", sorted(scored_df["month"].dropna().unique()))

    age_min, age_max = int(scored_df["age"].min()), int(scored_df["age"].max())
    f_age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    sidebar_filters = {
        "job": f_job, "marital": f_marital, "education": f_education,
        "housing": f_housing, "loan": f_loan, "contact": f_contact, "month": f_month,
    }
    filtered_df = scored_df.copy()
    for col, selected in sidebar_filters.items():
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    filtered_df = filtered_df[filtered_df["age"].between(f_age_range[0], f_age_range[1])]

    st.sidebar.divider()
    st.sidebar.write("**Model:** Tuned XGBoost (no-duration)")
    st.sidebar.write(f"**Decision threshold:** {threshold:.2f}")
    st.sidebar.caption("Epsilon AI ML Program")

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

    # ---- Seasonality & prior campaign outcome ----
    st.markdown("<div class='section-header'>\U0001F4C5 Predicted Probability by Month & Prior Outcome</div>", unsafe_allow_html=True)
    if len(filtered_df) > 0:
        month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        m1, m2 = st.columns(2)
        with m1:
            month_rate = (
                filtered_df.groupby("month")["predicted_prob"].mean()
                .reindex(month_order).dropna().reset_index(name="avg_predicted_prob")
            )
            fig_month = px.bar(
                month_rate, x="month", y="avg_predicted_prob", title="Avg Predicted Probability by Month",
                text_auto=".1%", color="month", color_discrete_sequence=COLOR_SEQUENCE,
            )
            fig_month.update_layout(yaxis_tickformat=".0%", showlegend=False, xaxis={"categoryorder": "array", "categoryarray": month_order})
            st.plotly_chart(fig_month, use_container_width=True)
        with m2:
            prev_rate = (
                filtered_df.groupby("Previous_Campaign_Success")["predicted_prob"].mean()
                .reset_index(name="avg_predicted_prob")
            )
            prev_rate["Previous_Campaign_Success"] = prev_rate["Previous_Campaign_Success"].map({0: "No prior success", 1: "Prior success"})
            fig_prev = px.bar(
                prev_rate, x="Previous_Campaign_Success", y="avg_predicted_prob",
                title="Avg Predicted Probability by Previous Campaign Success",
                text_auto=".1%", color="Previous_Campaign_Success", color_discrete_sequence=COLOR_SEQUENCE,
            )
            fig_prev.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig_prev, use_container_width=True)
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
