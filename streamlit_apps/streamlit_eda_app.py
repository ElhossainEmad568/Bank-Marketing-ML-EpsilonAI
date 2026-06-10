import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Bank Marketing EDA", layout="wide")

# Shared color theme for charts
PRIMARY_COLOR = "#1F6FEB"
COLOR_SEQUENCE = ["#1F6FEB", "#2EC4B6", "#F4A259", "#9B5DE5", "#E63946", "#0B2545"]
SUBSCRIPTION_COLORS = {"yes": "#2EC4B6", "no": "#0B2545"}

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


@st.cache_data
def load_data():
    cleaned_path = "cleaned_bank_dataset.csv"
    raw_path = "bank/bank-full.csv"
    try:
        df = pd.read_csv(cleaned_path)
        data_source = "Cleaned dataset"
    except FileNotFoundError:
        df = pd.read_csv(raw_path, sep=";")
        data_source = "Raw dataset"
    return df, data_source


@st.cache_data
def load_raw_data():
    return pd.read_csv("bank/bank-full.csv", sep=";")


df, data_source = load_data()

# ---- Sidebar: filters ----
st.sidebar.markdown("## \U0001F4CB Dashboard Filters")
st.sidebar.divider()

st.sidebar.markdown("**Demographics**")
job = st.sidebar.multiselect("Job", sorted(df["job"].dropna().unique()))
marital = st.sidebar.multiselect("Marital", sorted(df["marital"].dropna().unique()))
education = st.sidebar.multiselect("Education", sorted(df["education"].dropna().unique()))

st.sidebar.markdown("**Financial**")
housing = st.sidebar.multiselect("Housing Loan", sorted(df["housing"].dropna().unique()))
loan = st.sidebar.multiselect("Personal Loan", sorted(df["loan"].dropna().unique()))

st.sidebar.markdown("**Campaign**")
contact = st.sidebar.multiselect("Contact Type", sorted(df["contact"].dropna().unique()))
month = st.sidebar.multiselect("Month", sorted(df["month"].dropna().unique()))

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

filters = {
    "job": job, "marital": marital, "education": education,
    "housing": housing, "loan": loan, "contact": contact, "month": month,
}
filtered_df = df.copy()
for col, selected in filters.items():
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]
filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])]

st.sidebar.divider()
st.sidebar.caption("Epsilon AI ML Program")
st.sidebar.caption(f"{len(df):,} Customer Records")

# ---- Hero banner ----
st.markdown(
    f"""
    <div class="hero">
        <h1>\U0001F3E6 Bank Marketing EDA Dashboard</h1>
        <p>Exploratory Data Analysis &middot; {len(df):,} Customers &middot; {df.shape[1]} Features &middot;
        Data source: {data_source}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- KPI row ----
kpis = [
    (f"{len(filtered_df):,}", "Customers"),
    (f"{(filtered_df['y'].eq('yes').mean() * 100):.1f}%", "Subscription Rate"),
    (f"{filtered_df['balance'].mean():.0f}", "Avg Balance"),
    (f"{filtered_df['campaign'].mean():.1f}", "Avg Campaign Contacts"),
    (f"{filtered_df['duration'].mean():.0f}s", "Avg Call Duration"),
    (f"{(filtered_df['poutcome'].eq('success').mean() * 100):.1f}%", "Prior Success Rate"),
]
for col, (value, label) in zip(st.columns(6), kpis):
    col.markdown(
        f"<div class='kpi-card'><div class='value'>{value}</div><div class='label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

# ---- Data quality ----
st.markdown("<div class='section-header'>\U0001F50D Data Quality</div>", unsafe_allow_html=True)
unknown_counts = (df == "unknown").sum().sort_values(ascending=False)
qc1, qc2, qc3 = st.columns(3)
qc1.metric("Total Rows", f"{len(df):,}")
qc2.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
qc3.metric("Total Unknowns", f"{int(unknown_counts.sum()):,}")
st.dataframe(
    unknown_counts[unknown_counts > 0].reset_index().rename(columns={"index": "column", 0: "unknown_count"})
)

if data_source == "Cleaned dataset":
    st.caption("'unknown' values were already replaced with missing values during cleaning. Counts before cleaning, from the raw dataset:")
    raw_unknown_counts = (load_raw_data() == "unknown").sum().sort_values(ascending=False)
    st.dataframe(
        raw_unknown_counts[raw_unknown_counts > 0].reset_index().rename(columns={"index": "column", 0: "unknown_count"})
    )

# ---- Distributions ----
st.markdown("<div class='section-header'>\U0001F4CA Subscription & Distributions</div>", unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
with d1:
    fig_sub = px.pie(
        filtered_df, names="y", title="Subscription Outcome",
        color="y", color_discrete_map=SUBSCRIPTION_COLORS,
    )
    fig_sub.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_sub, use_container_width=True)
with d2:
    fig_balance = px.histogram(
        filtered_df, x="balance", nbins=50, title="Balance Distribution",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    st.plotly_chart(fig_balance, use_container_width=True)
with d3:
    fig_age = px.histogram(
        filtered_df, x="age", nbins=40, title="Age Distribution",
        color_discrete_sequence=[COLOR_SEQUENCE[1]],
    )
    st.plotly_chart(fig_age, use_container_width=True)

# ---- Subscription by segment ----
st.markdown("<div class='section-header'>\U0001F3AF Subscription by Segment</div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    job_rate = filtered_df.groupby("job")["y"].apply(lambda s: (s == "yes").mean()).reset_index(name="subscription_rate")
    fig_job = px.bar(
        job_rate, x="job", y="subscription_rate", title="Subscription Rate by Job",
        text_auto=".1%", color="job", color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig_job.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_job, use_container_width=True)
with s2:
    edu_rate = filtered_df.groupby("education")["y"].apply(lambda s: (s == "yes").mean()).reset_index(name="subscription_rate")
    fig_edu = px.bar(
        edu_rate, x="education", y="subscription_rate", title="Subscription Rate by Education",
        text_auto=".1%", color="education", color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig_edu.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_edu, use_container_width=True)
with s3:
    contact_rate = filtered_df.groupby("contact")["y"].apply(lambda s: (s == "yes").mean()).reset_index(name="subscription_rate")
    fig_contact = px.bar(
        contact_rate, x="contact", y="subscription_rate", title="Subscription Rate by Contact Type",
        text_auto=".1%", color="contact", color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig_contact.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_contact, use_container_width=True)

# ---- Correlations & feature importance ----
st.markdown("<div class='section-header'>\U0001F517 Correlations & Feature Importance</div>", unsafe_allow_html=True)
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
c1, c2 = st.columns(2)
with c1:
    fig_corr = px.imshow(
        filtered_df[num_cols].corr(), text_auto=".2f", aspect="auto", title="Correlation Heatmap",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
with c2:
    feature_importance_path = "saved_transformers/feature_importance.csv"
    if pd.io.common.file_exists(feature_importance_path):
        fi_df = pd.read_csv(feature_importance_path)
        fig_fi = px.bar(
            fi_df.head(10), x="importance", y="feature", title="Top Feature Importances",
            text_auto=".3f", orientation="h", color="importance", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance file not found. Run the ML notebook to generate it.")

# ---- Key insights ----
st.markdown("<div class='section-header'>\U0001F4A1 Key Insights</div>", unsafe_allow_html=True)
if len(filtered_df) > 0:
    best_job = job_rate.sort_values("subscription_rate", ascending=False).head(1)
    best_edu = edu_rate.sort_values("subscription_rate", ascending=False).head(1)
    best_contact = contact_rate.sort_values("subscription_rate", ascending=False).head(1)
    st.write(
        f"- Highest subscription job segment: {best_job.iloc[0]['job']} ({best_job.iloc[0]['subscription_rate']:.2%}).\n"
        f"- Highest subscription education segment: {best_edu.iloc[0]['education']} ({best_edu.iloc[0]['subscription_rate']:.2%}).\n"
        f"- Best performing contact type: {best_contact.iloc[0]['contact']} ({best_contact.iloc[0]['subscription_rate']:.2%}).\n"
        "- Previous campaign success is a strong positive indicator."
    )
else:
    st.write("No data available for the selected filters.")
