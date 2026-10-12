# 🏦 Customer Segmentation for Retail Banking
### End-to-End Data Science Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-green)](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Business Problem

Banks that treat all customers identically waste marketing budget and miss revenue opportunities. A one-size-fits-all approach leads to:

- **Low campaign conversion rates** (generic messaging doesn't resonate)
- **Preventable churn** (at-risk customers go unnoticed until it's too late)
- **Misallocated resources** (premium service wasted on low-value accounts)

**This project segments 45,000+ retail bank customers into actionable groups** using unsupervised machine learning, enabling the marketing and strategy teams to personalise every interaction.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| Records | 45,211 customers |
| Features | 17 (demographic + campaign history) |
| Target (for validation) | Term deposit subscription (y/n) |

**Key features used:** Age, account balance, job type, marital status, education, housing/personal loans, campaign contacts, previous contact history.

---

## 🛠️ Technical Approach

```
Raw Data  →  Cleaning  →  Feature Engineering  →  Scaling  →  Clustering  →  Interpretation
```

### 1. Feature Engineering (6 new behavioural features)

| Feature | Logic | Business Signal |
|---------|-------|-----------------|
| `balance_tier` | Cut into 5 bands | Wealth proxy |
| `contact_intensity` | log(campaign_contacts) | Engagement pressure |
| `financial_stress` | Sum of default + housing + loan flags | Debt burden |
| `prev_contacted` | pdays ≠ -1 | History with bank |
| `age_group` | 5 generational bands | Life stage |
| `high_engagement` | duration > median | Call quality signal |

### 2. Optimal K Selection

Used two complementary methods to choose k:

- **Elbow Method** — locate the point of diminishing inertia reduction
- **Silhouette Score** — maximise inter-cluster separation
- Final choice: **k = 4** (strong elbow + silhouette peak)

### 3. Clustering

Primary model: **K-Means** (k=4, n_init=10, random_state=42)

Compared against:
- Hierarchical Clustering (Ward linkage)
- DBSCAN (eps=1.5, min_samples=20)

K-Means was selected for its interpretability, computational efficiency on 45k+ rows, and superior silhouette score.

### 4. Visualisation

PCA (2 components) used to project high-dimensional cluster space into 2D for interpretation. Components explain ~38% of variance — sufficient for visual validation of cluster separation.

---

## 🧠 Cluster Profiles & Interpretation

| Cluster | Label | Avg Balance | Sub Rate | Size | Key Signal |
|---------|-------|------------|----------|------|------------|
| 0 | 👑 **High-Value VIPs** | €5,200+ | ~32% | ~18% | High balance, high engagement |
| 1 | ⚠️ **At-Risk / Dormant** | €200 | ~5% | ~30% | Low balance, multi-contact, no conversion |
| 2 | 🔵 **Core Stable** | €1,800 | ~12% | ~38% | Moderate balance, consistent |
| 3 | 🎯 **Occasional Movers** | €3,100 | ~15% | ~14% | Infrequent contact, decent balance |

> ⚡ **Key Insight:** The top 20% of customers by balance hold approximately **65% of total deposits** — identifying and retaining this group is the single highest-ROI action available to the bank.

---

## 💼 Business Recommendations

### 👑 Cluster 0 — High-Value VIPs
- **Strategy:** Premium relationship banking
- **Actions:** Dedicated relationship manager, wealth management products, priority support SLA, exclusive events
- **KPI:** Increase AUM per customer by 20%
- **Priority:** 🔴 Immediate

### ⚠️ Cluster 1 — At-Risk / Dormant
- **Strategy:** Retention & win-back campaigns
- **Actions:** Fee waivers, simplified product offers, personalised outreach before full churn
- **KPI:** Reduce churn by 15% in 6 months
- **Priority:** 🔴 Immediate

### 🔵 Cluster 2 — Core Stable
- **Strategy:** Cross-sell & loyalty
- **Actions:** Loyalty points, savings account nudges, credit upgrade offers at salary credit events
- **KPI:** Increase product holdings per customer 1.5×
- **Priority:** 🟡 Medium term

### 🎯 Cluster 3 — Occasional Movers
- **Strategy:** Trigger-based outreach only
- **Actions:** Contact at life events (large deposits, salary spikes), don't over-contact
- **KPI:** Triple engagement touchpoints
- **Priority:** 🟢 Long term

---

## 🚀 Running the Project

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-banking.git
cd customer-segmentation-banking
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook notebooks/customer_segmentation_banking.ipynb
```
The notebook auto-downloads the UCI dataset — no manual setup needed.

### 3. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```
Features:
- Adjustable k slider (live re-clustering)
- Age & balance filters
- PCA cluster map with centroids
- Segment profile heatmap
- Subscription rate by cluster
- Customer explorer (browse by segment)
- Per-cluster business recommendations

---

## 📁 Project Structure

```
customer-segmentation-banking/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── customer_segmentation_banking.ipynb   # Full analysis (Steps 1–12)
│
├── app/
│   └── streamlit_app.py                       # Interactive dashboard
│
├── data/
|   ├── raw/                                    # Auto-downloaded; .gitignore'd
|   └── proceessed/                           
|        ├── customers_segmented.csv        # Customer data + cluster labels
|        └── cluster_summary.csv            # Aggregated cluster stats
│
├── models/
│   ├── kmeans_model.pkl                       # Trained K-Means model
│   ├── scaler.pkl                             # Fitted StandardScaler
│   └── pca.pkl                               # Fitted PCA transformer
│
└── visuals/
    ├── 01_distributions.png
    ├── 02_categoricals.png
    ├── 03_balance_by_job.png
    ├── 04_correlation.png
    ├── 05_optimal_k.png
    ├── 06_silhouette_plot.png
    ├── 07_pca_clusters.png
    ├── 08_cluster_profiles.png
    ├── 09_business_metrics.png
    └── 10_model_comparison.png
```

---

## 🔬 Model Evaluation

| Model | Clusters Found | Silhouette Score | Notes |
|-------|---------------|-----------------|-------|
| **K-Means (k=4)** | 4 | **~0.18** | ✅ Best balance of interpretability + separation |
| Hierarchical (k=4) | 4 | ~0.16 | Slightly lower cohesion |
| DBSCAN (eps=1.5) | 3–5 | ~0.12 | Identifies noise but less stable |

*Silhouette scores on tabular financial data are typically lower than image/text data — 0.15–0.25 is a strong result in this domain.*

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
joblib>=1.2.0
```

---

## 🤝 About This Project

This project was built to simulate a **real industry data science workflow** in retail banking — from business problem framing through to deployable recommendations. It demonstrates:

✅ Business-first thinking (not just model accuracy)  
✅ Feature engineering depth  
✅ Model selection with justification  
✅ Actionable cluster interpretation  
✅ Production-ready code structure  
✅ Interactive stakeholder-facing dashboard  

---

Dataset: UCI Machine Learning Repository*
