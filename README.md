# CCHS Health Analytics Suite

This suite is a professional consolidation of four research and development projects focused on the Canadian Community Health Survey (CCHS) dataset. It provides a structured environment for clinical risk prediction, wellbeing analysis, and business intelligence.

## Directory Structure

### 📊 `Dashboards/`
Contains Power BI (`.pbix`) files and visual exports capturing key population health insights and demographic trends.

### 📓 `Notebooks/`
Consolidated high-value Jupyter Notebooks:
- `Clinical_Risk_Analysis.ipynb`: Detailed pipeline for health risk modeling.
- `Wellbeing_Analysis.ipynb`: Analysis and modeling of life satisfaction determinants.
- `Survey_Predictive_Pipeline.ipynb`: Reproducible end-to-end ML pipeline.

### 🤖 `Models/`
Trained machine learning artifacts, including:
- Health Risk Ensemble (XGBoost/Stacking).
- Life Satisfaction Predictors (LightGBM/GBR).
- Preprocessing templates and model metadata.

### 🚀 `App/`
A unified Streamlit-based Clinical Decision Support System:
- **Clinical Risk:** Real-time health risk assessments.
- **Wellbeing:** Life satisfaction scoring.
- **Analytics:** Population-level risk and wellbeing trends.

### 📁 `Data/`
The core CCHS dataset used across all analyses.

### 📄 `Docs/`
Documentation, including:
- Data Dictionaries and Mapping Guides.
- Project reports and research summaries.

## How to Get Started

### To Run the Clinical App:
1. Navigate to the `App/` directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run: `streamlit run app.py`.

### To Explore Notebooks:
Open any file in the `Notebooks/` directory using Jupyter Lab or VS Code to review the data science workflows.
