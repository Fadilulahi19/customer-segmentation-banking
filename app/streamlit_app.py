"""
Customer Segmentation Dashboard — Retail Banking
================================================
Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings, io, urllib.request, zipfile

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background-color: #0d0f14; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }

.metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #12151e 100%);
    border: 1px solid #2a2d3a;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #00d4ff;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.segment-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.stSelectbox > div > div { background-color: #1a1d27; }
.stSidebar { background-color: #0d0f14; border-right: 1px solid #1e2130; }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b"]
CLUSTER_NAMES  = ["High-Value VIPs", "At-Risk / Dormant", "Core Stable", "Occasional Movers"]
CLUSTER_EMOJI  = ["👑", "⚠️", "🔵", "🎯"]

# ─────────────────────────────────────────────
# DATA LOADING & PROCESSING (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(n_clusters: int = 4):
    # Load
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    response = urllib.request.urlopen(url)
    zf = zipfile.ZipFile(io.BytesIO(response.read()))
    with zf.open("bank-full.csv") as f:
        df = pd.read_csv(f, sep=";")

    # Clean
    for col in ["job", "education", "contact", "poutcome"]:
        mode_val = df[col][df[col] != "unknown"].mode()[0]
        df[col] = df[col].replace("unknown", mode_val)
    df.drop_duplicates(inplace=True)
    for col in ["default", "housing", "loan", "y"]:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # Feature engineering
    df["balance_tier"] = pd.cut(
        df["balance"], bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["contact_intensity"] = np.log1p(df["campaign"])
    df["prev_contacted"]    = (df["pdays"] != -1).astype(int)
    df["days_since_contact"] = df["pdays"].replace(-1, 999)
    df["age_group"]         = pd.cut(df["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df["financial_stress"]  = df["default"] + df["housing"] + df["loan"]
    df["high_engagement"]   = (df["duration"] > df["duration"].median()).astype(int)

    le = LabelEncoder()
    for col in ["job", "marital", "education", "contact", "poutcome", "month"]:
        df[col + "_enc"] = le.fit_transform(df[col])

    features = ["age", "balance", "balance_tier", "contact_intensity", "prev_contacted",
                "days_since_contact", "age_group", "financial_stress", "high_engagement",
                "job_enc", "marital_enc", "education_enc", "housing", "loan", "previous"]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return df, X_scaled, X_pca, km, pca

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Segmentation Controls")
    st.markdown("---")
    n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
    st.markdown("---")
    st.markdown("**Filter Customers**")
    age_range   = st.slider("Age Range",     min_value=18, max_value=95, value=(18, 95))
    bal_range   = st.slider("Balance (€)",   min_value=-8000, max_value=102000, value=(-8000, 102000), step=1000)
    st.markdown("---")
    show_cluster = st.multiselect(
        "Show Clusters",
        options=list(range(n_clusters)),
        default=list(range(n_clusters)),
        format_func=lambda x: f"Cluster {x}"
    )

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading & processing UCI Bank Marketing dataset…"):
    df, X_scaled, X_pca, km, pca = load_and_process(n_clusters)

# Apply filters
mask = (
    df["age"].between(*age_range) &
    df["balance"].between(*bal_range) &
    df["cluster"].isin(show_cluster)
)
df_filtered = df[mask].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏦 Customer Segmentation")
st.markdown("**Retail Banking · UCI Bank Marketing Dataset · K-Means Clustering**")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
sil_score  = silhouette_score(X_scaled, df["cluster"])
total_bal  = df_filtered["balance"].clip(lower=0).sum()
sub_rate   = df_filtered["y"].mean() * 100
top_pct    = df_filtered.nlargest(int(len(df_filtered) * 0.2), "balance")["balance"].sum()
top_share  = (top_pct / df_filtered["balance"].clip(lower=0).sum() * 100) if total_bal > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    (f"{len(df_filtered):,}",          "Customers"),
    (f"€{total_bal/1e6:.1f}M",         "Total Balance"),
    (f"{sub_rate:.1f}%",               "Subscription Rate"),
    (f"{sil_score:.3f}",               "Silhouette Score"),
    (f"{top_share:.0f}%",              "Balance: Top 20%"),
]
for col, (val, label) in zip([col1, col2, col3, col4, col5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📍 Cluster Map", "📊 Segment Profiles", "🔍 Customer Explorer", "💼 Recommendations"])

# ── TAB 1: CLUSTER MAP ────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### PCA Cluster Projection")
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")

        colors_use = CLUSTER_COLORS[:n_clusters]
        for i in range(n_clusters):
            m = df["cluster"] == i
            label = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"Cluster {i}"
            ax.scatter(X_pca[m, 0], X_pca[m, 1], c=colors_use[i],
                       alpha=0.4, s=6, label=label, edgecolors="none")

        centroids_pca = pca.transform(km.cluster_centers_)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c="white", marker="X", s=180, zorder=10, label="Centroids")

        exp = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({exp[0]:.1f}% variance)", color="#9ca3af", fontsize=10)
        ax.set_ylabel(f"PC2 ({exp[1]:.1f}% variance)", color="#9ca3af", fontsize=10)
        ax.tick_params(colors="#4b5563")
        ax.spines[:].set_color("#2a2d3a")
        leg = ax.legend(fontsize=9, framealpha=0.2, labelcolor="white",
                        facecolor="#1a1d27", edgecolor="#2a2d3a")
        ax.set_title("Customer Segments — 2D PCA", color="white", fontsize=13, pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("### Cluster Sizes")
        counts = df_filtered["cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        fig2.patch.set_facecolor("#0d0f14")
        ax2.set_facecolor("#0d0f14")
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=[f"C{i}" for i in counts.index],
            autopct="%1.1f%%",
            colors=colors_use,
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(linewidth=2, edgecolor="#0d0f14"),
        )
        for t in texts + autotexts:
            t.set_color("white")
            t.set_fontsize(10)
        ax2.set_title("Distribution", color="white", fontsize=11)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("### Silhouette Score")
        fig3, ax3 = plt.subplots(figsize=(4.5, 2))
        fig3.patch.set_facecolor("#0d0f14")
        ax3.set_facecolor("#0d0f14")
        ax3.barh(["Score"], [sil_score], color="#00d4ff", height=0.5)
        ax3.barh(["Score"], [1.0], color="#1a1d27", height=0.5)
        ax3.barh(["Score"], [sil_score], color="#00d4ff", height=0.5)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("0 → 1 (higher = better)", color="#6b7280", fontsize=9)
        ax3.tick_params(colors="white", labelsize=9)
        ax3.spines[:].set_color("#2a2d3a")
        ax3.text(sil_score + 0.02, 0, f"{sil_score:.3f}", color="#00d4ff",
                 va="center", fontsize=11, fontweight="bold")
        st.pyplot(fig3, use_container_width=True)
        plt.close()

# ── TAB 2: SEGMENT PROFILES ───────────────────────────────────────────────────
with tab2:
    profile_cols = ["age", "balance", "campaign", "previous", "financial_stress",
                    "high_engagement", "prev_contacted"]

    summary = df.groupby("cluster")[profile_cols].mean().round(2)
    summary["n"] = df.groupby("cluster").size()
    summary["sub_rate_%"] = (df.groupby("cluster")["y"].mean() * 100).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Cluster Profile Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")

        data = summary[profile_cols].T
        norm = (data - data.min(axis=1).values[:, None]) / \
               (data.max(axis=1) - data.min(axis=1) + 1e-9).values[:, None]

        sns.heatmap(norm, annot=data.round(1), fmt="g", cmap="Blues",
                    ax=ax, linewidths=0.8, linecolor="#0d0f14",
                    xticklabels=[f"C{i}" for i in range(n_clusters)],
                    cbar_kws={"shrink": 0.6})
        ax.set_title("Feature Profiles (Normalised)", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#2a2d3a")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("### Subscription Rate by Segment")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")

        sub = summary["sub_rate_%"]
        bars = ax.bar(range(n_clusters), sub.values,
                      color=CLUSTER_COLORS[:n_clusters], width=0.5, edgecolor="#0d0f14")
        for bar, val in zip(bars, sub.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", color="white", fontsize=11, fontweight="bold")

        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f"C{i}" for i in range(n_clusters)], color="white")
        ax.set_ylabel("Subscription Rate (%)", color="#9ca3af")
        ax.set_title("Term Deposit Conversion by Cluster", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2a2d3a")
        ax.set_facecolor("#0d0f14")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("### Full Cluster Summary Table")
    st.dataframe(
        summary.style.background_gradient(cmap="Blues", subset=profile_cols),
        use_container_width=True
    )

# ── TAB 3: CUSTOMER EXPLORER ──────────────────────────────────────────────────
with tab3:
    st.markdown("### Browse Customers by Segment")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_cluster = st.selectbox(
            "Select Cluster",
            options=list(range(n_clusters)),
            format_func=lambda x: f"Cluster {x} — {CLUSTER_NAMES[x] if x < len(CLUSTER_NAMES) else ''}"
        )
    with col_f2:
        sort_by = st.selectbox("Sort by", ["balance", "age", "campaign", "previous", "financial_stress"])

    cluster_df = df_filtered[df_filtered["cluster"] == selected_cluster].copy()
    cluster_df = cluster_df.sort_values(sort_by, ascending=False)

    display_cols = ["age", "job", "marital", "education", "balance",
                    "housing", "loan", "campaign", "previous", "financial_stress", "y"]
    show_df = cluster_df[display_cols].head(200).copy()
    show_df.columns = ["Age", "Job", "Marital", "Education", "Balance (€)",
                       "Housing Loan", "Personal Loan", "Contacts", "Prev. Contacts",
                       "Financial Stress", "Subscribed"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in Cluster", f"{len(cluster_df):,}")
    c2.metric("Avg Balance", f"€{cluster_df['balance'].mean():,.0f}")
    c3.metric("Subscription Rate", f"{cluster_df['y'].mean()*100:.1f}%")

    st.dataframe(show_df, use_container_width=True, height=400)
    st.caption(f"Showing top 200 of {len(cluster_df):,} customers in this cluster (sorted by {sort_by})")

# ── TAB 4: RECOMMENDATIONS ────────────────────────────────────────────────────
with tab4:
    st.markdown("### 💼 Business Recommendations by Segment")

    recs = [
        {
            "label": "High-Value VIPs", "color": "#00d4ff", "emoji": "👑",
            "profile": "High balance · Engaged · High subscription rate",
            "action": "Premium relationship banking. Offer wealth management products, dedicated relationship manager, priority customer support.",
            "kpi": "Increase AUM per customer by 20%",
            "priority": "🔴 Immediate",
            "channels": "Direct RM call · Private banking portal · Exclusive events"
        },
        {
            "label": "At-Risk / Dormant", "color": "#ff6b6b", "emoji": "⚠️",
            "profile": "Low/negative balance · Multiple contact attempts · No conversion",
            "action": "Targeted retention campaign. Win-back offers, fee waivers, simplified product bundles. Act before full churn.",
            "kpi": "Reduce churn by 15% in 6 months",
            "priority": "🔴 Immediate",
            "channels": "Email win-back series · SMS with offer · In-app push"
        },
        {
            "label": "Core Stable", "color": "#51cf66", "emoji": "🔵",
            "profile": "Moderate balance · Consistent engagement · Reliable",
            "action": "Loyalty programs & cross-sell. Introduce savings accounts, credit upgrades, auto-invest nudges at salary credit events.",
            "kpi": "Increase products held per customer 1.5×",
            "priority": "🟡 Medium term",
            "channels": "Mobile app · Monthly newsletter · In-branch"
        },
        {
            "label": "Occasional Movers", "color": "#ffd43b", "emoji": "🎯",
            "profile": "Moderate-high balance · Infrequent contact · Low prior engagement",
            "action": "Trigger-based outreach. Contact at life events — large deposits, salary spikes, birthday milestones. Don't over-contact.",
            "kpi": "Increase engagement touchpoints 3×",
            "priority": "🟢 Long term",
            "channels": "Triggered email · Contextual in-app banner"
        },
    ]

    for i, rec in enumerate(recs[:n_clusters]):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1d27 0%, #12151e 100%);
                    border: 1px solid #2a2d3a; border-left: 4px solid {rec['color']};
                    border-radius: 10px; padding: 20px 24px; margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                <span style="font-size:1.5rem">{rec['emoji']}</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
                             font-weight:700; color:{rec['color']}">Cluster {i} — {rec['label']}</span>
                <span style="margin-left:auto; font-size:0.8rem; color:#9ca3af">{rec['priority']}</span>
            </div>
            <div style="color:#9ca3af; font-size:0.85rem; margin-bottom:10px">📊 {rec['profile']}</div>
            <div style="color:#e5e7eb; font-size:0.95rem; margin-bottom:10px">🎯 <strong>Strategy:</strong> {rec['action']}</div>
            <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:8px">
                <span style="background:#0d0f14; border:1px solid #2a2d3a; border-radius:4px;
                             padding:4px 12px; font-size:0.8rem; color:#6ee7b7">📈 {rec['kpi']}</span>
                <span style="background:#0d0f14; border:1px solid #2a2d3a; border-radius:4px;
                             padding:4px 12px; font-size:0.8rem; color:#93c5fd">📡 {rec['channels']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Key Takeaway")
    st.info("Typically, the **top 20% of customers by balance hold 60–70% of total deposits**. "
            "Protecting and growing this VIP cohort is the single highest-ROI action available to retail banking strategy teams.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4b5563; font-size:0.8rem; font-family:IBM Plex Mono,monospace'>"
    "Customer Segmentation · UCI Bank Marketing Dataset · K-Means + PCA · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
