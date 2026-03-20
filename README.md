# 🩺 CCHS Health Analytics Suite — Predictive Clinical Intelligence from Canadian Survey Data

> End-to-end machine learning platform that transforms 50+ health indicators from the Canadian Community Health Survey into actionable clinical risk scores and wellbeing predictions, powered by ensemble ML models and an interactive Streamlit application.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189FDD?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-02569B?style=for-the-badge)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboards-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [📖 Usage](#-usage)
- [📓 Notebooks](#-notebooks)
- [📊 Power BI Dashboards](#-power-bi-dashboards)
- [🧠 Models & Artifacts](#-models--artifacts)
- [📐 Data Dictionary](#-data-dictionary)
- [🚀 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [👤 Author](#-author)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

The **CCHS Health Analytics Suite** is a professional consolidation of four research and development projects built on the [Canadian Community Health Survey (CCHS)](https://www.statcan.gc.ca/en/survey/household/3226) dataset. It combines rigorous exploratory data analysis, advanced feature engineering, and production-grade machine learning pipelines into a unified platform for **clinical risk prediction**, **life satisfaction modeling**, and **population health intelligence**.

This project is designed for data scientists, public health researchers, and clinical decision-support teams who need to transform raw survey data into interpretable, deployable health predictions.

---

## ✨ Features

- 🏥 **Clinical Risk Assessment Engine** — Real-time health risk scoring using a stacked ensemble of XGBoost, LightGBM, and scikit-learn classifiers trained on 52 engineered features
- 🌟 **Wellbeing & Life Satisfaction Predictor** — Regression-based life satisfaction scoring (0–10 scale) using gradient boosted models with behavioral and socioeconomic inputs
- 📊 **Interactive Clinical Dashboard** — Streamlit-powered dashboard with patient metrics, risk distributions, and wellbeing trend visualizations
- 👤 **Patient Management System** — Full CRUD patient registry with SQLite-backed persistent storage, linked to assessment histories
- 🔐 **Role-Based Authentication** — SHA-256 hashed login system with auto-provisioned admin account and doctor role management
- 📈 **Population Health Analytics** — KDE-smoothed risk score distributions and box-plot wellbeing analysis across the patient population
- 🧪 **Reproducible ML Pipelines** — Three Jupyter notebooks covering end-to-end workflows from data cleaning through hyperparameter-tuned model evaluation
- 📉 **Power BI Dashboards** — Interactive business intelligence reports with multi-page demographic and health trend visualizations
- 🔄 **Automated Feature Engineering** — Domain-driven features including `Age_BP_Risk` interaction, `Comorbidity_Score` aggregation, and `Activity_Normalized` ratios
- ✅ **Input Validation & Error Handling** — Server-side validation of all clinical inputs with range checks and type enforcement before model inference

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core language for all ML, data, and app logic |
| **Web Framework** | Streamlit | Interactive clinical decision-support application |
| **ML — Gradient Boosting** | XGBoost, LightGBM | High-performance classifiers for risk and satisfaction |
| **ML — Framework** | scikit-learn | Preprocessing pipelines, stacking ensemble, evaluation metrics |
| **ML — Imbalanced Data** | imbalanced-learn | SMOTE and resampling for class imbalance in health outcomes |
| **ML — Tuning** | Optuna | Bayesian hyperparameter optimization |
| **ML — Explainability** | SHAP | Feature importance and model interpretability |
| **Data Processing** | Pandas, NumPy | Data manipulation, cleaning, and feature engineering |
| **Visualization** | Matplotlib, Seaborn | Statistical plots, EDA charts, and in-app analytics |
| **Database** | SQLite | Lightweight patient registry and assessment history storage |
| **BI Dashboards** | Power BI | Multi-page demographic and health trend reports |
| **Serialization** | Joblib | Model persistence and artifact management |

---

## 📁 Project Structure

```
Canadian-Community-Health-Survey-Predictive-Analytics/
│
├── App/                                # Streamlit Clinical Decision Support System
│   ├── app.py                          # Main application — 5-page clinical platform
│   ├── auth_manager.py                 # Authentication with SHA-256 password hashing
│   ├── database_manager.py             # SQLite ORM for patients, assessments, users
│   ├── model_logic.py                  # Input validation and model preprocessing logic
│   └── requirements.txt                # Python dependencies for the app
│
├── Dashboards/                         # Power BI Business Intelligence
│   ├── Project.pbix                    # Primary Power BI dashboard file
│   ├── Project .pbix                   # Alternate dashboard version
│   └── Screenshot 2025-12-08 *.png     # Dashboard page screenshots (5 pages)
│
├── Data/
│   └── health_dataset.csv              # Core CCHS dataset (~14 MB, 50+ columns)
│
├── Docs/                               # Documentation & Research
│   ├── Data_dictionary.txt             # Column-by-column field descriptions
│   ├── Health_Dataset_Decoding_and_mapping_dictionary.docx
│   ├── Gemini.md                       # Project methodology and design notes
│   ├── 12308126 Savinay.pdf            # Academic project report
│   └── document.pdf                    # Supplementary documentation
│
├── Models/                             # Trained ML Artifacts
│   ├── best_health_ensemble.pkl        # Stacked ensemble classifier (~8 MB)
│   ├── best_xgb_model.pkl              # XGBoost pipeline with preprocessor
│   ├── lgbm_low_satisfaction_clf.pkl    # LightGBM satisfaction classifier
│   ├── artifact_best_model_with_meta.pkl # Wellbeing model + feature metadata
│   ├── pipeline_life_satisfaction.pkl  # Life satisfaction regression pipeline
│   ├── template_df.pkl                 # Feature template for input alignment
│   └── app_metadata.json              # Feature lists, thresholds, baseline values
│
├── Notebooks/                          # Jupyter Analysis Pipelines
│   ├── Clinical_Risk_Analysis.ipynb    # Health risk modeling end-to-end
│   ├── Wellbeing_Analysis.ipynb        # Life satisfaction determinant analysis
│   └── Survey_Predictive_Pipeline.ipynb # Reproducible ML pipeline template
│
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

```
Python >= 3.10
pip (Python package manager)
Git
```

Optional for dashboards:
```
Power BI Desktop (Windows only) — for .pbix files
Jupyter Lab / VS Code — for notebook exploration
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/savinaysingh7/Canadian-Community-Health-Survey-Predictive-Analytics.git
cd Canadian-Community-Health-Survey-Predictive-Analytics

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r App/requirements.txt
```

### Running the Application

```bash
# Navigate to the App directory and launch
cd App
streamlit run app.py
```

The app will open at `http://localhost:8501`. Log in with the default credentials:

| Field | Value |
|---|---|
| **Username** | `admin` |
| **Password** | `admin123` |

### Exploring Notebooks

```bash
# From the project root
jupyter lab Notebooks/
```

Open any of the three notebooks to review the full data science workflow — from data cleaning and EDA through model training and evaluation.

---

## 📖 Usage

### Clinical Risk Assessment

1. Navigate to **Patients** → add a new patient with name, age group, and gender
2. Go to **Clinical Risk** → select the patient
3. Fill in vitals (age, BMI, blood pressure, cholesterol), mental health indicators (general health, stress, satisfaction), and lifestyle factors (smoking, activity, diet, sleep apnea)
4. Click **Predict Clinical Risk** — the ensemble model returns a probability score and HIGH/LOW classification based on the optimal threshold (0.5)

```python
# Example of how the risk prediction works internally
input_data = {
    'Age': 4,                          # 50-64 age group
    'BMI_18_above': 2,                 # Overweight/Obese
    'High_BP': 1.0,                    # Yes
    'High_cholestrol': 1.0,            # Yes
    'Gen_health_state': 4,             # Fair
    'Mental_health_state': 3,          # Good
    'Stress_level': 4,                 # High
    'Life_satisfaction': 5.0,          # Moderate
    'Smoked': 15.0,                    # 15 cigarettes/day
    'Physical_vigorous_act_time': 30.0,# 30 min/week
    'Fruit_veg_con': 1,               # Low consumption
    'Sleep_apnea': 1.0                 # Diagnosed
}

# Model pipeline: validate → align to template → preprocess → ensemble predict
errors = validate_risk_input(input_data)
input_df = prepare_risk_input(input_data, template_df, xgb_pipeline)
probability = ensemble_model.predict_proba(input_df)[0][1]  # → 0.78 (HIGH RISK)
```

### Wellbeing Prediction

1. Go to **Wellbeing Check** → select a patient
2. Input values for each wellbeing feature (the model uses the features stored in `artifact_best_model_with_meta.pkl`)
3. Click **Predict Life Satisfaction** — returns a score on a 0–10 scale

### Analytics Dashboard

The **Analytics** page provides two visualizations:
- **Clinical Risk Trends** — KDE-smoothed histogram of all risk scores across the patient population
- **Wellbeing Distribution** — Box plot showing the spread and outliers of life satisfaction scores

---

## 📓 Notebooks

### `Clinical_Risk_Analysis.ipynb`
End-to-end clinical risk modeling pipeline: data cleaning with column-by-column inspection, EDA with 50+ visual and statistical analyses, feature engineering (comorbidity scoring, age-risk interactions), and multi-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking Ensemble).

### `Wellbeing_Analysis.ipynb`
Deep analysis of life satisfaction determinants: correlation analysis across socioeconomic and health variables, regression modeling with gradient boosted regressors, and feature importance ranking via SHAP values.

### `Survey_Predictive_Pipeline.ipynb`
Reproducible ML pipeline template: standardized preprocessing with `ColumnTransformer`, Optuna-based hyperparameter tuning, cross-validated evaluation, and artifact serialization for deployment.

---

## 📊 Power BI Dashboards

The `Dashboards/` directory contains interactive Power BI reports with multiple pages covering:

- Population demographics and provincial distribution
- Health condition prevalence (chronic diseases, mental health, pain)
- Lifestyle factor analysis (smoking, alcohol, physical activity)
- Socioeconomic correlates (income, education, food security)
- Cross-tabulated health outcomes by gender, age group, and region

Open the `.pbix` files in [Power BI Desktop](https://powerbi.microsoft.com/desktop/) to explore the interactive reports.

---

## 🧠 Models & Artifacts

| Artifact | Type | Size | Description |
|---|---|---|---|
| `best_health_ensemble.pkl` | Stacking Classifier | ~8 MB | Production ensemble combining XGBoost + LightGBM + base learners for clinical risk |
| `best_xgb_model.pkl` | XGBoost Pipeline | ~500 KB | Full sklearn pipeline with preprocessor, used for feature alignment and transformation |
| `lgbm_low_satisfaction_clf.pkl` | LightGBM Classifier | ~1.7 MB | Binary classifier for low life satisfaction detection |
| `artifact_best_model_with_meta.pkl` | Model + Metadata | ~270 KB | Wellbeing regression model bundled with its required feature list |
| `pipeline_life_satisfaction.pkl` | Regression Pipeline | ~270 KB | Full life satisfaction prediction pipeline |
| `template_df.pkl` | DataFrame Template | ~3 KB | Feature schema template for input alignment during inference |
| `app_metadata.json` | JSON Config | ~4 KB | Feature lists (52 features), categorical/numeric splits, optimal threshold (0.5), healthy baseline values |

---

## 📐 Data Dictionary

The core dataset (`Data/health_dataset.csv`) contains **50+ columns** from the Canadian Community Health Survey. Key variable groups:

| Category | Variables | Examples |
|---|---|---|
| **Demographics** | 8 columns | `Age`, `Gender`, `Marital_status`, `Province`, `Aboriginal_identity` |
| **Health Status** | 10 columns | `Gen_health_state`, `Mental_health_state`, `Health_utility_index`, `Pain_status` |
| **Chronic Conditions** | 8 columns | `High_BP`, `Diabetic`, `Cardiovascular_con`, `Mood_disorder`, `Anxiety_disorder` |
| **Lifestyle** | 9 columns | `Smoked`, `Weekly_alcohol`, `Cannabis_use`, `Fruit_veg_con`, `Physical_vigorous_act_time` |
| **Socioeconomic** | 7 columns | `Edu_level`, `Total_income`, `Working_status`, `Food_security`, `Insurance_cover` |
| **Engineered** | 5 columns | `Age_BP_Risk`, `Smoked_Cholesterol`, `Comorbidity_Score`, `Activity_Normalized`, `Age_numeric` |

Full descriptions are available in [`Docs/Data_dictionary.txt`](Docs/Data_dictionary.txt) and the decoding guide in [`Docs/Health_Dataset_Decoding_and_mapping_dictionary.docx`](Docs/Health_Dataset_Decoding_and_mapping_dictionary.docx).

---

## 🚀 Deployment

### Local Development

```bash
cd App
streamlit run app.py
# Opens at http://localhost:8501
```

### Streamlit Community Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and set:
   - **Main file path:** `App/app.py`
   - **Requirements file:** `App/requirements.txt`
4. Deploy — the app will be live at `https://<your-app>.streamlit.app`

> **Note:** Ensure the `Models/` directory with all `.pkl` artifacts is committed to the repository, as the app loads them at runtime via relative paths.

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r App/requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "App/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t cchs-health-analytics .
docker run -p 8501:8501 cchs-health-analytics
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and test thoroughly
# 4. Commit with a descriptive message
git commit -m "feat: add new risk factor to clinical model"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Areas for contribution:**
- Additional ML models (neural networks, survival analysis)
- Enhanced UI/UX for the Streamlit app
- Unit and integration test coverage
- API layer for external system integration
- Multi-language support for the clinical interface

---

## 👤 Author

**Savinay Singh**
- GitHub: [@savinaysingh7](https://github.com/savinaysingh7)

---

## 🙏 Acknowledgments

- [**Statistics Canada**](https://www.statcan.gc.ca/) — Canadian Community Health Survey (CCHS) dataset
- [**Streamlit**](https://streamlit.io/) — Rapid prototyping framework for the clinical application
- [**XGBoost**](https://xgboost.readthedocs.io/) & [**LightGBM**](https://lightgbm.readthedocs.io/) — Gradient boosting libraries powering the ensemble models
- [**scikit-learn**](https://scikit-learn.org/) — ML pipeline infrastructure and preprocessing
- [**SHAP**](https://shap.readthedocs.io/) — Model explainability and feature importance analysis
- [**Optuna**](https://optuna.org/) — Hyperparameter optimization framework
- [**Power BI**](https://powerbi.microsoft.com/) — Business intelligence dashboards

---

<p align="center">
  <i>Built with ❤️ for public health research and clinical decision support</i>
</p>
