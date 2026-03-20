# 🧠 CCHS Health Data Science Playground

> An end-to-end interactive ML platform for predictive analytics on Canadian population health data.

![Version](https://img.shields.io/badge/version-5.1-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CCHS%20Public--Use-lightgrey)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Workflow Guide](#-workflow-guide)
- [Architecture](#-architecture)
- [Dataset & Metadata](#-dataset--metadata)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🌟 Overview

The **CCHS Health Data Science Playground** is a fully interactive, browser-based machine learning application built with Streamlit, purpose-built for analyzing the **Canadian Community Health Survey (CCHS)** dataset. It provides a guided five-stage scientific workflow — from raw data ingestion through to a deployability decision report — with zero code required from the end user.

The platform is designed for data scientists, health researchers, and policy analysts who want to rapidly prototype and validate predictive models on population health variables (e.g., predicting `Health_utility_index`, `Stress_level`, or `Life_satisfaction`) while enforcing rigorous standards around data leakage, model stability, and feature dependency.

---

## ✨ Features

### 📥 Data Ingestion
- Auto-loads `health_dataset.csv` and `Data_dictionary.txt` on startup using `@st.cache_data` for fast repeat access
- **Quality Advisor** automatically flags columns with >40% missingness, numeric-as-categorical mismatches, and mixed-type object columns

### 🧹 Dictionary-Driven Data Cleaning
- **Method-Aware Auto-Clean** replaces CCHS-specific "not stated" / "don't know" sentinel codes (e.g., `97`, `98`, `99`, `9996`–`9999`) with `NaN` across 19 columns based on the official CCHS public-use documentation
- **Full cleaning audit report** showing per-column new missing values introduced and total rows affected
- **Dataset version history** with one-click rollback to any prior state (Raw, Method-Cleaned, or custom)

### 📊 Exploratory Data Analysis
- **Univariate Analysis** with metadata-aware plotting: categorical columns render decoded human-readable labels (e.g., `1 → "Male"`, `5 → "Extremely stressful"`); numeric columns render KDE-annotated histograms
- **Bivariate Correlation Bridge** computing Pearson correlations of all numeric features against any chosen target, visualised as a ranked bar chart (top 10)
- In-line column descriptions sourced from the data dictionary on every plot

### 🔬 Experimental Lab (Modeling)
- **Five regression algorithms** benchmarked head-to-head: Baseline (Mean), Linear Regression, Decision Tree (max_depth=5), Random Forest (100 estimators), and Gradient Boosting
- **Metadata-aware ordinal encoding**: ordinal features (e.g., `Gen_health_state`, `Total_income`) are encoded using the ground-truth CCHS category order, not arbitrary alphabetical order
- **Spearman monotonicity guard**: warns if a feature marked Ordinal has a Spearman correlation < 0.1 with the target, requiring explicit user confirmation before proceeding
- **Data leakage detector** flagging features with >0.95 Pearson correlation with the target or ID-like columns with row-unique cardinality
- **Multi-seed stability check** (seeds 42, 52, 62) reporting R² standard deviation across random splits to quantify result reliability
- **Ablation / Dependency Quantification** via permutation importance: identifies the single most important feature and retrains without it to measure the R² drop
- **Error diagnostics panel**: residuals distribution (KDE histogram) and residuals-vs-predicted scatter plot for the champion model
- **One-click experiment reproducibility**: reload any prior experiment's full configuration (target, features, encoding strategy, model selection) for re-runs
- **Champion model export** as a fitted `sklearn.Pipeline` `.pkl` file via `joblib`

### 📝 Decision Report
- Auto-generates a structured Markdown report covering: Executive Summary, Decision Readiness Verdict (pass/fail with specific risk codes), Methodology, Data Source Notes, and a timestamped Project Activity Log
- **Three-tier risk system**: 🔴 CRITICAL (leakage detected), 🟠 Unstable (R² Std > 0.05), 🟡 High Dependency (ablation R² drop > 0.1)
- **Feature Role Annotation**: classify each model feature as `Descriptive`, `Actionable`, or `Sensitive` for ethics and governance documentation
- Download report as `.md` file
- **Persistent experiment store**: all experiments serialised to `experiments.json` and reloaded across sessions

---

## 🛠️ Tech Stack

| Category | Library | Version | Purpose |
|---|---|---|---|
| UI Framework | Streamlit | Latest | Interactive web application shell |
| Data Manipulation | Pandas | Latest | DataFrame operations, cleaning, EDA |
| Numerical Computing | NumPy | Latest | Array math, imputation, statistics |
| Machine Learning | Scikit-Learn | Latest | Pipelines, preprocessing, regressors, evaluation |
| Statistical Testing | SciPy | Latest | Spearman correlation for ordinal validation |
| Visualisation | Seaborn | Latest | Count plots, histograms, bar charts |
| Visualisation | Matplotlib | Latest | Figure/axes management |
| Model Persistence | Joblib | Latest | `.pkl` champion model serialisation |
| State / Logging | Python stdlib (`json`, `uuid`, `time`, `dataclasses`) | 3.8+ | Experiment logging and session management |

---

## 📁 Project Structure

```
Canadian-Community-Health-Survey-Predictive-Analytics/
├── app.py                                        # Main Streamlit application (all logic & UI)
├── health_dataset.csv                            # Modified CCHS public-use dataset
├── Data_dictionary.txt                           # Column descriptions (Markdown table format)
├── Health_Dataset_Decoding_and_mapping_dictionary.docx  # Full CCHS variable encoding guide
├── ok.ipynb                                      # Supplementary Jupyter notebook for ad-hoc analysis
├── experiments.json                              # Auto-generated experiment log (created at runtime)
└── README.md                                     # This file
```

> **Note:** `experiments.json` is created automatically on the first model run and persists all experiment history across Streamlit sessions.

---

## ⚡ Quick Start

### Prerequisites

```
Python >= 3.8
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/savinaysingh7/Canadian-Community-Health-Survey-Predictive-Analytics.git
cd Canadian-Community-Health-Survey-Predictive-Analytics

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install streamlit pandas numpy seaborn matplotlib scikit-learn scipy joblib

# 4. Confirm required files are in the root directory
#    health_dataset.csv
#    Data_dictionary.txt

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

---

## 🗺️ Workflow Guide

The sidebar radio nav drives a strict left-to-right scientific pipeline. Each stage feeds the next.

### Stage 1 — 📥 Load
Navigate to **Load** in the sidebar. The dataset and data dictionary are automatically ingested. A **Quality Advisor** panel surfaces any structural issues (high missingness, type mismatches) before you proceed.

### Stage 2 — 🧹 Clean
Click **✨ Auto-Clean (Apply Dictionary Rules)**. This replaces all CCHS sentinel "not answered" codes with `NaN` according to the official documentation. A per-column audit table shows exactly what changed. Use the **History** panel to restore any prior version if needed.

### Stage 3 — 📊 EDA
- Use **Univariate Analysis** to inspect any column's distribution. Toggle **Decode Labels** to see human-readable category names.
- Expand **Bivariate Analysis** to identify the top 10 features correlated with any potential target — use this to inform your feature selection in Stage 4.

### Stage 4 — 🔬 Lab (Model)
1. Select a **Target Variable** (must be numeric, e.g. `Health_utility_index`)
2. Select **Features** from the multiselect
3. Assign **Ordinal Features** — the app pre-suggests CCHS-documented ordinals and validates them with Spearman
4. Choose **Algorithms** and optionally enable the **3-seed Stability Check**
5. Click **🚀 Run Experiment**

Results include a ranked metrics table (R², MAE, RMSE, Std), a dependency quantification summary, error diagnostic plots, and a `.pkl` download of the champion model.

### Stage 5 — 📝 Report
Select an experiment from the dropdown. Optionally annotate feature roles (`Descriptive`, `Actionable`, `Sensitive`). The app generates and renders a full Decision Report with a pass/fail deployment verdict. Download it as Markdown.

---

## 🏗️ Architecture

The application follows a **single-file stateful Streamlit architecture** with three utility classes and a persistent dataclass layer:

```
┌─────────────────────────────────────────────────────┐
│                    app.py                           │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  DataUtils   │  │  ModelUtils  │  │ReportUtils│  │
│  │  ──────────  │  │  ──────────  │  │──────────│  │
│  │ load_data()  │  │get_regressors│  │generate_ │  │
│  │ load_dict()  │  │train_and_    │  │markdown()│  │
│  │ check_quality│  │  evaluate()  │  │          │  │
│  │ check_leakage│  │analyze_errors│  │          │  │
│  │ validate_    │  │              │  │          │  │
│  │  ordinal()   │  │              │  │          │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  st.session_state (in-memory state)         │    │
│  │  df_raw / df_active / versions / history /  │    │
│  │  experiments / project_log / data_dict      │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  experiments.json  (disk persistence)       │    │
│  │  Experiment dataclass → JSON serialisation  │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**Key design decisions:**
- `@st.cache_data` on data loading prevents re-reading from disk on every widget interaction
- `sklearn.Pipeline` wraps the full preprocessing + model chain so the exported `.pkl` is self-contained and deployable without external preprocessing code
- `ColumnTransformer` dynamically constructs separate paths for numeric (StandardScaler), nominal (OneHotEncoder), and ordinal (OrdinalEncoder with ground-truth category lists) features
- Permutation importance is computed post-hoc on the test set to avoid train-set bias in the ablation analysis

---

## 📊 Dataset & Metadata

| Property | Detail |
|---|---|
| Source | Statistics Canada — Canadian Community Health Survey (CCHS) |
| File | `health_dataset.csv` |
| Columns | 51 health, demographic, and behavioural variables |
| Key variable types | Self-reported health outcomes, chronic conditions, lifestyle behaviours, socioeconomic indicators |
| Encoding | Numeric codes per CCHS public-use documentation (decoded via `Value_Maps` and `Data_dictionary.txt`) |
| Invalid code handling | 19 columns have CCHS-specific sentinel codes replaced with `NaN` during the Clean stage |

For full variable descriptions, see `Data_dictionary.txt`. For encoding and mapping logic, see `Health_Dataset_Decoding_and_mapping_dictionary.docx`.

---

## 🤝 Contributing

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your-username>/Canadian-Community-Health-Survey-Predictive-Analytics.git
cd Canadian-Community-Health-Survey-Predictive-Analytics

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and test locally
streamlit run app.py

# 5. Commit with a descriptive message
git commit -m "feat: add classification support to Lab stage"

# 6. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Suggested contribution areas:**
- Classification model support (logistic regression, SVC) alongside the existing regression suite
- SHAP-based feature importance as an alternative to permutation importance
- Additional EDA plots (correlation heatmap, pairplot for selected features)
- Export results to PDF or Excel in the Report stage

---

## 👤 Author

**Savinay Singh**
[GitHub @savinaysingh7](https://github.com/savinaysingh7)

---

## 🙏 Acknowledgments

- **Statistics Canada** for making the CCHS public-use microdata file available for research
- **Streamlit** for the reactive application framework
- **Scikit-Learn** for the composable pipeline and model ecosystem
- **SciPy** for statistical validation utilities