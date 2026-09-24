"""
Mall Customer Intelligence Dashboard
=====================================
A complete Streamlit BI application for mall customer segmentation,
spending analysis, and strategic insights.

Dataset: Mall_Customers.csv
Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fb; }
    
    /* KPI card */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1d3461;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #57606a;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #2d8a4e;
        margin-top: 2px;
    }
    
    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1d3461;
        border-left: 4px solid #3b82d4;
        padding-left: 10px;
        margin-bottom: 12px;
    }
    
    /* Insight box */
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1f2328;
    }
    .risk-box {
        background: #fff4f0;
        border-left: 4px solid #e05c3a;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1f2328;
    }
    .opportunity-box {
        background: #f0fff4;
        border-left: 4px solid #2d8a4e;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1f2328;
    }
    .action-box {
        background: #fdfaf0;
        border-left: 4px solid #c49b0a;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1f2328;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1d3461;
    }
    [data-testid="stSidebar"] * {
        color: #e8edf5 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #c5d0e0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    /* Table */
    .dataframe { font-size: 0.85rem !important; }
    
    /* Segment badge */
    .seg-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")

@st.cache_data(show_spinner=False)
def load_data():
    """Load and clean the mall customers dataset."""
    df = pd.read_csv(DATA_PATH)

    # Standardise column names for easier access
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Annual Income (k$)": "Annual_Income_k",
        "Spending Score (1-100)": "Spending_Score",
    }, inplace=True)

    # Drop duplicates (none expected, but defensive)
    df.drop_duplicates(subset="CustomerID", keep="first", inplace=True)

    # Correct dtypes
    df["Age"] = df["Age"].astype(int)
    df["Annual_Income_k"] = df["Annual_Income_k"].astype(float)
    df["Spending_Score"] = df["Spending_Score"].astype(float)

    # Derived feature: Age group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 71],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )

    # Derived feature: Income tier
    income_q33 = df["Annual_Income_k"].quantile(0.33)
    income_q67 = df["Annual_Income_k"].quantile(0.67)
    df["Income_Tier"] = pd.cut(
        df["Annual_Income_k"],
        bins=[0, income_q33, income_q67, df["Annual_Income_k"].max() + 1],
        labels=["Low", "Mid", "High"],
    )

    # Derived feature: Spending tier
    df["Spending_Tier"] = pd.cut(
        df["Spending_Score"],
        bins=[0, 33, 66, 100],
        labels=["Low", "Mid", "High"],
    )

    return df


@st.cache_data(show_spinner=False)
def run_kmeans(df, k=5):
    """K-Means clustering on Annual Income & Spending Score."""
    features = df[["Annual_Income_k", "Spending_Score"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    # Compute silhouette score
    sil = silhouette_score(X_scaled, labels)

    df_out = df.copy()
    df_out["Cluster"] = labels

    # Build cluster profiles with meaningful names
    profiles = (
        df_out.groupby("Cluster")
        .agg(
            Count=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual_Income_k", "mean"),
            Avg_Spending=("Spending_Score", "mean"),
            Female_Pct=("Gender", lambda x: (x == "Female").mean() * 100),
        )
        .round(1)
    )

    # Assign segment names based on income/spending quadrant
    def name_cluster(row):
        inc = row["Avg_Income"]
        sp = row["Avg_Spending"]
        inc_med = df["Annual_Income_k"].median()
        sp_med = df["Spending_Score"].median()
        if inc >= inc_med and sp >= sp_med:
            return "High-Value Loyalists"
        elif inc >= inc_med and sp < sp_med:
            return "Affluent Savers"
        elif inc < inc_med and sp >= sp_med:
            return "Budget Enthusiasts"
        elif inc < inc_med and sp < sp_med:
            return "Low-Engagement Group"
        else:
            return "Average Shoppers"

    profiles["Segment_Name"] = profiles.apply(name_cluster, axis=1)

    # Handle ties by appending cluster id to duplicates
    seen = {}
    new_names = []
    for idx, row in profiles.iterrows():
        name = row["Segment_Name"]
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name} {seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    profiles["Segment_Name"] = new_names

    cluster_map = profiles["Segment_Name"].to_dict()
    df_out["Segment"] = df_out["Cluster"].map(cluster_map)

    return df_out, profiles, sil


@st.cache_data(show_spinner=False)
def elbow_data(df, max_k=10):
    """Compute inertia for elbow chart."""
    features = df[["Annual_Income_k", "Spending_Score"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return list(range(2, max_k + 1)), inertias


# ─────────────────────────────────────────────
# KPI COMPUTATION
# ─────────────────────────────────────────────
def compute_kpis(df):
    total = len(df)
    female_pct = (df["Gender"] == "Female").mean() * 100
    male_pct = 100 - female_pct
    avg_age = df["Age"].mean()
    avg_income = df["Annual_Income_k"].mean()
    avg_spend = df["Spending_Score"].mean()
    median_income = df["Annual_Income_k"].median()
    median_spend = df["Spending_Score"].median()

    # High-value customers: above-median on both income AND spending
    high_value_pct = (
        (df["Annual_Income_k"] >= median_income) & (df["Spending_Score"] >= median_spend)
    ).mean() * 100

    # At-risk: high income but low spending (untapped potential)
    at_risk_pct = (
        (df["Annual_Income_k"] >= median_income) & (df["Spending_Score"] < median_spend)
    ).mean() * 100

    # Correlation income vs spending
    corr = df["Annual_Income_k"].corr(df["Spending_Score"])

    return {
        "total_customers": total,
        "female_pct": round(female_pct, 1),
        "male_pct": round(male_pct, 1),
        "avg_age": round(avg_age, 1),
        "avg_income": round(avg_income, 1),
        "avg_spend": round(avg_spend, 1),
        "median_income": round(median_income, 1),
        "median_spend": round(median_spend, 1),
        "high_value_pct": round(high_value_pct, 1),
        "at_risk_pct": round(at_risk_pct, 1),
        "corr_income_spend": round(corr, 3),
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.markdown(
        "## 🏪 Mall Customer\n### Intelligence",
        unsafe_allow_html=False,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "👥 Customer Demographics",
            "💳 Spending Analysis",
            "🎯 Customer Segments",
            "⚠️ Risk & Opportunity",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
    gender_filter = st.sidebar.selectbox("Gender", gender_options)

    age_group_options = ["All"] + [str(g) for g in df["Age_Group"].cat.categories.tolist()]
    age_group_filter = st.sidebar.multiselect("Age Group(s)", age_group_options[1:], default=[])

    income_min = int(df["Annual_Income_k"].min())
    income_max = int(df["Annual_Income_k"].max())
    income_range = st.sidebar.slider(
        "Annual Income (k$)", income_min, income_max, (income_min, income_max)
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Dataset: Mall_Customers.csv | 200 customers | 5 features")

    # Apply filters
    filtered = df.copy()
    if gender_filter != "All":
        filtered = filtered[filtered["Gender"] == gender_filter]
    if age_group_filter:
        filtered = filtered[filtered["Age_Group"].astype(str).isin(age_group_filter)]
    filtered = filtered[
        (filtered["Annual_Income_k"] >= income_range[0]) &
        (filtered["Annual_Income_k"] <= income_range[1])
    ]

    return page, filtered


# ─────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────
def page_executive_overview(df, df_seg, profiles, kpis):
    st.markdown("## 📊 Executive Overview")
    st.markdown("_Key performance indicators and strategic summary derived from actual customer data._")

    # ── KPI Row ──────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_items = [
        (k1, str(kpis["total_customers"]), "Total Customers", ""),
        (k2, f"{kpis['avg_income']}k", "Avg Annual Income", "USD"),
        (k3, f"{kpis['avg_spend']}", "Avg Spending Score", "/ 100"),
        (k4, f"{kpis['high_value_pct']}%", "High-Value Customers", "High income + high spend"),
        (k5, f"{kpis['at_risk_pct']}%", "Affluent Savers", "High income, low spend"),
    ]
    for col, val, label, delta in kpi_items:
        with col:
            st.markdown(
                f'<div class="kpi-card"><p class="kpi-value">{val}</p>'
                f'<p class="kpi-label">{label}</p>'
                f'<p class="kpi-delta">{delta}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two charts row ────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Income vs Spending Score</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df_seg,
            x="Annual_Income_k",
            y="Spending_Score",
            color="Segment",
            hover_data=["Gender", "Age"],
            labels={"Annual_Income_k": "Annual Income (k$)", "Spending_Score": "Spending Score"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=380,
        )
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Segment Distribution</p>', unsafe_allow_html=True)
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig2 = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=380,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(
            paper_bgcolor="#ffffff",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Insights ─────────────────────────────
    st.markdown('<p class="section-header">Key Executive Findings</p>', unsafe_allow_html=True)

    # Dynamically computed insights
    best_seg = profiles.loc[profiles["Avg_Spending"].idxmax(), "Segment_Name"]
    best_seg_spend = profiles["Avg_Spending"].max()
    best_seg_income = profiles.loc[profiles["Avg_Spending"].idxmax(), "Avg_Income"]
    best_seg_count = profiles.loc[profiles["Avg_Spending"].idxmax(), "Count"]

    worst_seg = profiles.loc[profiles["Avg_Spending"].idxmin(), "Segment_Name"]
    worst_spend = profiles["Avg_Spending"].min()
    worst_income = profiles.loc[profiles["Avg_Spending"].idxmin(), "Avg_Income"]

    female_spend = df[df["Gender"] == "Female"]["Spending_Score"].mean()
    male_spend = df[df["Gender"] == "Male"]["Spending_Score"].mean()
    gender_delta = abs(female_spend - male_spend)
    higher_gender = "Female" if female_spend > male_spend else "Male"

    young_spend = df[df["Age"] <= 35]["Spending_Score"].mean()
    older_spend = df[df["Age"] > 35]["Spending_Score"].mean()

    corr = kpis["corr_income_spend"]

    insights = [
        f"🏆 <b>Top segment '{best_seg}'</b> ({best_seg_count} customers) averages a spending score of "
        f"<b>{best_seg_spend:.1f}</b> with avg income <b>${best_seg_income:.0f}k</b> — the highest-revenue driver.",
        f"📉 <b>Segment '{worst_seg}'</b> earns <b>${worst_income:.0f}k</b> avg income yet scores only "
        f"<b>{worst_spend:.1f}</b> in spending — representing untapped wallet share.",
        f"👩 <b>{higher_gender} customers</b> spend <b>{gender_delta:.1f} points</b> more on average "
        f"(Female: {female_spend:.1f}, Male: {male_spend:.1f}) — gender-targeted promotions can improve ROI.",
        f"🎂 <b>Customers aged ≤35</b> score <b>{young_spend:.1f}</b> vs <b>{older_spend:.1f}</b> for >35 — "
        f"younger shoppers are significantly more active spenders.",
        f"📊 <b>Income–Spending correlation is {corr:.3f}</b> — near-zero, meaning income alone does NOT "
        f"predict spending; psychographic and behavioral factors dominate.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — CUSTOMER DEMOGRAPHICS
# ─────────────────────────────────────────────
def page_demographics(df):
    st.markdown("## 👥 Customer Demographics")
    st.markdown("_Demographic breakdown of the filtered customer base._")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Gender Distribution</p>', unsafe_allow_html=True)
        gender_counts = df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(
            gender_counts, names="Gender", values="Count",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            height=320,
        )
        fig.update_traces(textinfo="percent+label", pull=[0.05, 0])
        fig.update_layout(paper_bgcolor="#ffffff", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Age Group Distribution</p>', unsafe_allow_html=True)
        age_counts = df["Age_Group"].value_counts().sort_index().reset_index()
        age_counts.columns = ["Age_Group", "Count"]
        age_counts["Age_Group"] = age_counts["Age_Group"].astype(str)
        fig2 = px.bar(
            age_counts, x="Age_Group", y="Count",
            color="Count", color_continuous_scale="Blues",
            labels={"Age_Group": "Age Group", "Count": "# Customers"},
            height=320,
        )
        fig2.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">Age Distribution by Gender</p>', unsafe_allow_html=True)
        fig3 = px.histogram(
            df, x="Age", color="Gender", nbins=20, barmode="overlay",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            opacity=0.75, height=320,
        )
        fig3.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">Annual Income Distribution</p>', unsafe_allow_html=True)
        fig4 = px.histogram(
            df, x="Annual_Income_k", color="Gender", nbins=20, barmode="overlay",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            labels={"Annual_Income_k": "Annual Income (k$)"},
            opacity=0.75, height=320,
        )
        fig4.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Age group vs avg income table
    st.markdown('<p class="section-header">Age Group × Income & Spending Profile</p>', unsafe_allow_html=True)
    tbl = (
        df.groupby("Age_Group", observed=True)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Income=("Annual_Income_k", "mean"),
            Avg_Spending=("Spending_Score", "mean"),
            Female_Pct=("Gender", lambda x: round((x == "Female").mean() * 100, 1)),
        )
        .round(1)
        .reset_index()
    )
    tbl.columns = ["Age Group", "Customers", "Avg Income (k$)", "Avg Spending Score", "Female %"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 3 — SPENDING ANALYSIS
# ─────────────────────────────────────────────
def page_spending_analysis(df):
    st.markdown("## 💳 Spending Analysis")
    st.markdown("_What drives spending score? Visualised across income, age, and gender._")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Income vs Spending Score</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df, x="Annual_Income_k", y="Spending_Score",
            color="Gender", trendline="ols",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            labels={"Annual_Income_k": "Annual Income (k$)", "Spending_Score": "Spending Score"},
            height=360,
        )
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Age vs Spending Score</p>', unsafe_allow_html=True)
        fig2 = px.scatter(
            df, x="Age", y="Spending_Score",
            color="Gender", trendline="ols",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            labels={"Spending_Score": "Spending Score"},
            height=360,
        )
        fig2.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">Avg Spending Score by Age Group</p>', unsafe_allow_html=True)
        grp = (
            df.groupby(["Age_Group", "Gender"], observed=True)["Spending_Score"]
            .mean()
            .reset_index()
        )
        grp["Age_Group"] = grp["Age_Group"].astype(str)
        fig3 = px.bar(
            grp, x="Age_Group", y="Spending_Score", color="Gender", barmode="group",
            color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
            labels={"Age_Group": "Age Group", "Spending_Score": "Avg Spending Score"},
            height=340,
        )
        fig3.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">Spending Score Distribution by Income Tier</p>', unsafe_allow_html=True)
        df_tier = df.dropna(subset=["Income_Tier"])
        fig4 = px.box(
            df_tier, x="Income_Tier", y="Spending_Score", color="Income_Tier",
            color_discrete_map={"Low": "#f59e42", "Mid": "#3b82d4", "High": "#7c5cd8"},
            labels={"Income_Tier": "Income Tier", "Spending_Score": "Spending Score"},
            category_orders={"Income_Tier": ["Low", "Mid", "High"]},
            height=340,
        )
        fig4.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Driver summary table
    st.markdown('<p class="section-header">Spending Driver Summary</p>', unsafe_allow_html=True)
    drivers = []
    for gender in df["Gender"].unique():
        sub = df[df["Gender"] == gender]
        drivers.append({
            "Dimension": f"Gender: {gender}",
            "Group": gender,
            "Avg Spending Score": round(sub["Spending_Score"].mean(), 1),
            "Customers": len(sub),
            "Avg Income (k$)": round(sub["Annual_Income_k"].mean(), 1),
        })
    for ag in df["Age_Group"].cat.categories:
        sub = df[df["Age_Group"] == ag]
        if len(sub) > 0:
            drivers.append({
                "Dimension": f"Age Group: {ag}",
                "Group": str(ag),
                "Avg Spending Score": round(sub["Spending_Score"].mean(), 1),
                "Customers": len(sub),
                "Avg Income (k$)": round(sub["Annual_Income_k"].mean(), 1),
            })
    driver_df = pd.DataFrame(drivers).sort_values("Avg Spending Score", ascending=False)
    st.dataframe(driver_df[["Dimension", "Customers", "Avg Spending Score", "Avg Income (k$)"]],
                 use_container_width=True, hide_index=True)

    # Correlation insight
    corr_age = df["Age"].corr(df["Spending_Score"])
    corr_inc = df["Annual_Income_k"].corr(df["Spending_Score"])
    st.markdown(
        f'<div class="insight-box">📐 <b>Correlation Analysis:</b> '
        f'Age ↔ Spending Score = <b>{corr_age:.3f}</b> (moderate negative — older customers spend less). &nbsp;|&nbsp; '
        f'Income ↔ Spending Score = <b>{corr_inc:.3f}</b> (near-zero — income alone does not predict spending).</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE 4 — CUSTOMER SEGMENTS
# ─────────────────────────────────────────────
def page_segments(df, df_seg, profiles, sil):
    st.markdown("## 🎯 Customer Segments")
    st.markdown("_K-Means clustering (k=5) on Annual Income & Spending Score._")

    # Silhouette + cluster count
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">5</p><p class="kpi-label">Clusters (k)</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{sil:.3f}</p><p class="kpi-label">Silhouette Score</p><p class="kpi-delta">0 = random, 1 = perfect</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        largest_seg = profiles.loc[profiles["Count"].idxmax(), "Segment_Name"]
        largest_count = profiles["Count"].max()
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{largest_count}</p>'
            f'<p class="kpi-label">Largest Segment</p><p class="kpi-delta">{largest_seg}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<p class="section-header">Cluster Map</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df_seg,
            x="Annual_Income_k",
            y="Spending_Score",
            color="Segment",
            symbol="Gender",
            hover_data=["Age", "CustomerID"],
            labels={"Annual_Income_k": "Annual Income (k$)", "Spending_Score": "Spending Score"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=420,
        )
        # Add cluster centroids
        for _, row in profiles.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Avg_Income"]], y=[row["Avg_Spending"]],
                mode="markers+text",
                marker=dict(symbol="x", size=14, color="black"),
                text=[row["Segment_Name"].split()[0]],
                textposition="top center",
                showlegend=False,
            ))
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Elbow Curve</p>', unsafe_allow_html=True)
        ks, inertias = elbow_data(df)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color="#3b82d4", width=2),
            marker=dict(size=7),
        ))
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#e05c3a", annotation_text="k=5")
        fig_elbow.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            height=200, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown('<p class="section-header">Segment Sizes</p>', unsafe_allow_html=True)
        seg_bar = profiles[["Segment_Name", "Count"]].sort_values("Count", ascending=True)
        fig_bar = px.bar(
            seg_bar, x="Count", y="Segment_Name", orientation="h",
            color="Count", color_continuous_scale="Blues",
            labels={"Segment_Name": "", "Count": "Customers"},
            height=200,
        )
        fig_bar.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Segment profiles table
    st.markdown('<p class="section-header">Segment Profiles</p>', unsafe_allow_html=True)
    display_profiles = profiles.copy().reset_index(drop=True)
    display_profiles = display_profiles.rename(columns={
        "Segment_Name": "Segment",
        "Count": "Customers",
        "Avg_Age": "Avg Age",
        "Avg_Income": "Avg Income (k$)",
        "Avg_Spending": "Avg Spending Score",
        "Female_Pct": "Female %",
    })
    display_profiles = display_profiles[["Segment", "Customers", "Avg Age", "Avg Income (k$)", "Avg Spending Score", "Female %"]]
    st.dataframe(display_profiles, use_container_width=True, hide_index=True)

    # Per-segment recommendations
    st.markdown('<p class="section-header">Segment-Specific Recommendations</p>', unsafe_allow_html=True)
    seg_recs = {
        "High-Value Loyalists": "🎁 Priority VIP program — exclusive previews, loyalty points, dedicated concierge. Focus on retention.",
        "Affluent Savers": "💡 Needs-based marketing — showcase value, bundles, quality assurance. Convert saving mindset to spend.",
        "Budget Enthusiasts": "🛍️ Flash sales, discount events, cashback schemes. These customers WANT to spend — make it affordable.",
        "Low-Engagement Group": "📧 Re-engagement campaigns — email / SMS with seasonal offers. Low cost, modest recovery potential.",
        "Average Shoppers": "📈 Upselling strategy — cross-category promotions, 'buy more save more' to push above average spend.",
    }
    for _, row in profiles.iterrows():
        seg = row["Segment_Name"]
        # Match by prefix
        rec = next((v for k, v in seg_recs.items() if k.lower() in seg.lower()), "Personalised outreach recommended.")
        st.markdown(
            f'<div class="insight-box"><b>{seg}</b> ({int(row["Count"])} customers, '
            f'avg income ${row["Avg_Income"]:.0f}k, avg spend {row["Avg_Spending"]:.1f})<br>{rec}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE 5 — RISK & OPPORTUNITY
# ─────────────────────────────────────────────
def page_risk_opportunity(df, df_seg, profiles, kpis):
    st.markdown("## ⚠️ Risk & Opportunity Analysis")
    st.markdown("_Evidence-based risks, growth opportunities, and recommended strategic actions._")

    # ── Computed values for narrative ─────────
    affluent_savers = df_seg[df_seg["Segment"].str.contains("Affluent", case=False, na=False)]
    low_eng = df_seg[df_seg["Segment"].str.contains("Low-Engagement", case=False, na=False)]
    budget_enth = df_seg[df_seg["Segment"].str.contains("Budget", case=False, na=False)]
    high_val = df_seg[df_seg["Segment"].str.contains("High-Value", case=False, na=False)]

    aff_count = len(affluent_savers)
    aff_avg_income = affluent_savers["Annual_Income_k"].mean() if aff_count > 0 else 0
    aff_avg_spend = affluent_savers["Spending_Score"].mean() if aff_count > 0 else 0
    low_count = len(low_eng)
    bud_count = len(budget_enth)
    bud_avg_income = budget_enth["Annual_Income_k"].mean() if bud_count > 0 else 0
    bud_avg_spend = budget_enth["Spending_Score"].mean() if bud_count > 0 else 0
    hv_count = len(high_val)

    young_customers = df[df["Age"] <= 25]
    young_pct = len(young_customers) / len(df) * 100
    young_spend = young_customers["Spending_Score"].mean()

    female_pct = kpis["female_pct"]
    female_spend = df[df["Gender"] == "Female"]["Spending_Score"].mean()

    col_r, col_o = st.columns(2)

    with col_r:
        st.markdown('<p class="section-header">⚠️ Identified Risks</p>', unsafe_allow_html=True)
        risks = [
            (
                f"💼 <b>Affluent Underperformers ({aff_count} customers)</b> — "
                f"Avg income ${aff_avg_income:.0f}k yet spending score only {aff_avg_spend:.1f}/100. "
                f"Risk: these high-income shoppers are spending their budget at competing venues."
            ),
            (
                f"👴 <b>Older Customer Disengagement</b> — Customers aged >55 score "
                f"{df[df['Age'] > 55]['Spending_Score'].mean():.1f} vs overall mean {kpis['avg_spend']:.1f}. "
                f"Risk: aging demographic drift reducing total spend if not retained."
            ),
            (
                f"📉 <b>Low-Engagement Segment ({low_count} customers)</b> — "
                f"Avg income ${low_eng['Annual_Income_k'].mean() if low_count > 0 else 0:.0f}k, spending score "
                f"{low_eng['Spending_Score'].mean() if low_count > 0 else 0:.1f}/100. "
                f"Risk: churn to competitors with no retention strategy in place."
            ),
        ]
        for r in risks:
            st.markdown(f'<div class="risk-box">{r}</div>', unsafe_allow_html=True)

    with col_o:
        st.markdown('<p class="section-header">🚀 Growth Opportunities</p>', unsafe_allow_html=True)
        opps = [
            (
                f"💎 <b>Convert Affluent Savers to High-Value</b> — {aff_count} customers with avg income "
                f"${aff_avg_income:.0f}k are spending only {aff_avg_spend:.1f}/100. Closing 50% of this gap "
                f"would add ~{int(aff_count * (kpis['avg_spend'] - aff_avg_spend) * 0.5)} total spend-points."
            ),
            (
                f"🛍️ <b>Youth Monetisation (Age ≤25)</b> — {len(young_customers)} customers ({young_pct:.1f}% of base) "
                f"already scoring {young_spend:.1f} avg — youth-targeted events, pop-ups, and social campaigns "
                f"can capitalise on this high-energy segment."
            ),
            (
                f"👩 <b>Female Customer Deepening</b> — {female_pct:.0f}% of customers are female with avg spending score "
                f"{female_spend:.1f}. Personalised loyalty programs and category promotions targeting female shoppers "
                f"can grow repeat visits and basket size."
            ),
        ]
        for o in opps:
            st.markdown(f'<div class="opportunity-box">{o}</div>', unsafe_allow_html=True)

    # ── Recommended Actions ───────────────────
    st.markdown('<p class="section-header">✅ Recommended Actions</p>', unsafe_allow_html=True)
    actions = [
        (
            "1️⃣ <b>Launch a Tiered VIP Loyalty Program</b> — "
            f"Enrol the {hv_count} High-Value Loyalists immediately with exclusive benefits (early access, personal shoppers). "
            "Target: retain 95%+ of this segment and grow their avg spend by 10%."
        ),
        (
            f"2️⃣ <b>Activate Affluent Saver Re-engagement Campaign</b> — "
            f"Email + SMS outreach to {aff_count} affluent-income, low-spending customers. "
            "Offer curated premium bundles aligned to their income bracket. Target: +15 spending score points."
        ),
        (
            f"3️⃣ <b>Youth-Centric Events & Social Commerce</b> — "
            f"Host monthly in-mall events (pop-up shops, influencer activations) targeting ≤25 age group "
            f"({len(young_customers)} customers, avg spend {young_spend:.1f}). Target: +5% footfall from under-25 demographic."
        ),
        (
            "4️⃣ <b>Gender-Differentiated Promotions</b> — "
            f"Female shoppers ({female_pct:.0f}% of base) have higher spending scores. "
            "Develop category-specific promotions (fashion, beauty, lifestyle) to deepen wallet share."
        ),
        (
            f"5️⃣ <b>Win-Back Programme for Low-Engagement Segment</b> — "
            f"Re-engage {low_count} low-engagement customers through seasonal discount events, push notifications, "
            "and 'we miss you' vouchers. Even a +10-point spend lift adds measurable revenue."
        ),
    ]
    for a in actions:
        st.markdown(f'<div class="action-box">{a}</div>', unsafe_allow_html=True)

    # ── FACT → INSIGHT → OPPORTUNITY → ACTION ─
    st.markdown('<p class="section-header">FACT → INSIGHT → OPPORTUNITY → ACTION Framework</p>', unsafe_allow_html=True)
    fia_data = [
        {
            "Finding": "Finding 1: Near-Zero Income–Spend Correlation",
            "FACT": f"Pearson correlation between Annual Income and Spending Score = {kpis['corr_income_spend']:.3f}.",
            "INSIGHT": "Income is NOT a reliable predictor of spending. Psychographic segmentation outperforms demographic targeting.",
            "OPPORTUNITY": "Reallocate targeting budget from income-based to behaviour/cluster-based audiences.",
            "ACTION": "Use K-Means segment labels in CRM to replace broad income-bracket targeting with cluster-specific campaigns.",
        },
        {
            "Finding": "Finding 2: Budget Enthusiasts — High Spend, Low Income",
            "FACT": f"{bud_count} customers with avg income ${bud_avg_income:.0f}k achieve avg spending score {bud_avg_spend:.1f}/100.",
            "INSIGHT": "Desire to spend is not constrained by income. This segment responds to price-accessible offers.",
            "OPPORTUNITY": f"Grow Budget Enthusiast basket via value deals, BOGO offers, and instalment payment options.",
            "ACTION": "Create a dedicated 'Smart Shopper' loyalty tier with bonus points on discounted purchases.",
        },
        {
            "Finding": "Finding 3: Female Customer Majority",
            "FACT": f"{female_pct:.1f}% of customers are female. Female avg spending score = {female_spend:.1f}.",
            "INSIGHT": "Female shoppers are both the majority AND higher spenders — a dual leverage point.",
            "OPPORTUNITY": "Optimise store layout, promotions calendar, and brand partnerships toward female-led categories.",
            "ACTION": "Partner with 3 female-skewing brands for co-branded in-mall activations in Q1.",
        },
    ]
    for item in fia_data:
        with st.expander(item["Finding"], expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**📌 FACT**\n\n{item['FACT']}")
            c2.markdown(f"**💡 INSIGHT**\n\n{item['INSIGHT']}")
            c3.markdown(f"**🚀 OPPORTUNITY**\n\n{item['OPPORTUNITY']}")
            c4.markdown(f"**✅ ACTION**\n\n{item['ACTION']}")

    # ── Risk-Opportunity matrix chart ─────────
    st.markdown('<p class="section-header">Segment Risk-Opportunity Matrix</p>', unsafe_allow_html=True)
    matrix_data = profiles.copy()
    matrix_data["Risk_Score"] = 100 - matrix_data["Avg_Spending"]
    matrix_data["Opp_Score"] = matrix_data["Avg_Income"] - matrix_data["Avg_Spending"]
    fig_matrix = px.scatter(
        matrix_data,
        x="Avg_Income",
        y="Avg_Spending",
        size="Count",
        color="Segment_Name",
        text="Segment_Name",
        labels={"Avg_Income": "Avg Income (k$)", "Avg_Spending": "Avg Spending Score", "Segment_Name": "Segment"},
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=420,
    )
    fig_matrix.update_traces(textposition="top center")
    fig_matrix.add_hline(y=matrix_data["Avg_Spending"].mean(), line_dash="dot", line_color="grey")
    fig_matrix.add_vline(x=matrix_data["Avg_Income"].mean(), line_dash="dot", line_color="grey")
    fig_matrix.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fb",
        showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load data
    with st.spinner("Loading data…"):
        df_raw = load_data()
        df_seg_full, profiles, sil = run_kmeans(df_raw)

    # Sidebar with filters → returns filtered df
    page, df_filtered = render_sidebar(df_raw)

    # Re-run clustering on filtered subset only if subset is large enough
    if len(df_filtered) >= 10:
        df_seg, profiles_filtered, sil_f = run_kmeans(df_filtered)
    else:
        st.warning("Filter result has fewer than 10 customers — showing full dataset.")
        df_seg, profiles_filtered, sil_f = df_seg_full, profiles, sil

    kpis = compute_kpis(df_filtered)

    # Route pages
    if page == "📊 Executive Overview":
        page_executive_overview(df_filtered, df_seg, profiles_filtered, kpis)
    elif page == "👥 Customer Demographics":
        page_demographics(df_filtered)
    elif page == "💳 Spending Analysis":
        page_spending_analysis(df_filtered)
    elif page == "🎯 Customer Segments":
        page_segments(df_filtered, df_seg, profiles_filtered, sil_f)
    elif page == "⚠️ Risk & Opportunity":
        page_risk_opportunity(df_filtered, df_seg, profiles_filtered, kpis)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#57606a;font-size:0.78rem;'>"
        "Mall Customer Intelligence Dashboard · Built with Streamlit & Plotly · "
        "Data: Mall_Customers.csv (200 customers)</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
