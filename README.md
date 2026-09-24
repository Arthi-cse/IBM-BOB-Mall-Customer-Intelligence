# 🏪 Mall Customer Intelligence Dashboard

> A professional, end-to-end Business Intelligence application built with **Streamlit** and **Plotly** for mall customer segmentation, spending analysis, and strategic decision support.

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | Mall_Customers.csv |
| **Rows** | 200 customers |
| **Columns** | CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100) |
| **Domain** | Retail / Mall Analytics |
| **Business Problem** | Understand who mall customers are, what drives their spending, which segments deliver the most value, and where risks and growth opportunities lie |

---

## 🎯 Business Problem Statement

A shopping mall has collected demographic and spending data for 200 customers. The business needs to:
1. Identify distinct customer segments to personalise marketing
2. Understand what drives spending score (age? income? gender?)
3. Quantify high-value vs at-risk customer groups
4. Generate evidence-based strategic recommendations

---

## 📊 Key Findings

| Finding | Value |
|---------|-------|
| Total customers | 200 |
| Female share | 56.0% |
| Average Annual Income | $60.56k |
| Average Spending Score | 50.2 / 100 |
| Income ↔ Spending Correlation | ~0.01 (near-zero) |
| High-Value customers (high income + high spend) | ~25% |
| Affluent Savers (high income, low spend) | ~25% |

### 5 Key Insights
1. **Income does NOT predict spending** — Pearson correlation ≈ 0.01. Psychographic and behavioural factors dominate.
2. **High-Value Loyalists** are the top revenue segment with spending scores well above average.
3. **Affluent Savers** represent the largest untapped opportunity — high income, low engagement.
4. **Younger customers (≤35)** spend significantly more than older cohorts.
5. **Female shoppers** hold a 4–8 point spending score advantage over male shoppers.

---

## 📁 Project Structure

```
MallCustomerBI/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── ProjectReport.docx      ← Full analytical report
└── data/
    └── Mall_Customers.csv  ← Source dataset
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- pip package manager

### 2. Navigate to project directory
```bash
cd MallCustomerBI
```

### 3. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **📊 Executive Overview** | KPI cards, income vs spending scatter, segment distribution pie, 5 key executive insights |
| **👥 Customer Demographics** | Gender pie, age group bar chart, gender-age histogram, age × income table |
| **💳 Spending Analysis** | Income vs spending scatter with trendlines, age vs spending, spending by age group & income tier |
| **🎯 Customer Segments** | K-Means cluster map (k=5), elbow curve, silhouette score, segment profiles & recommendations |
| **⚠️ Risk & Opportunity** | 3 identified risks, 3 growth opportunities, 5 recommended actions, FACT→INSIGHT→OPPORTUNITY→ACTION framework |

---

## 🔧 Interactive Filters (Sidebar)

- **Gender** — Filter to Female / Male / All
- **Age Group(s)** — Multi-select age groups (18-25, 26-35, 36-45, 46-55, 56+)
- **Annual Income Range** — Slider from $15k to $137k

All charts, KPIs, and insights update dynamically based on active filters.

---

## 🤖 Methodology

### Customer Segmentation
- **Algorithm:** K-Means clustering (k=5)
- **Features:** Annual Income (k$), Spending Score (1-100)
- **Preprocessing:** StandardScaler normalization
- **Validation:** Silhouette Score, Elbow Method
- **Optimal k:** 5 (standard for mall customer analytics, validated by elbow)

### KPI Computation
All KPIs are computed dynamically from the filtered dataset — no hardcoded values.

### Driver Analysis
Pearson correlation between each feature and Spending Score; grouped statistics by gender, age group, and income tier.

---

## 📦 Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.36 |
| Visualisation | Plotly 5.22 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Machine Learning | scikit-learn 1.5 |
| Report Generation | python-docx 1.1 |
| Language | Python 3.10+ |

---

## 📄 License

This project is created for educational and analytical purposes.

---

*Mall Customer Intelligence Dashboard · Powered by Streamlit & Plotly*
