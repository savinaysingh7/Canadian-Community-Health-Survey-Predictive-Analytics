# Canadian Community Health Survey (CCHS) Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E)

A comprehensive data science playground built with Streamlit for exploring and applying predictive analytics on the Canadian Community Health Survey (CCHS) dataset. The application provides an interactive end-to-end machine learning workflow, right from data ingestion to model deployment readiness.

## 🧠 Application Features

This platform is divided into 5 core stages representing a scientific data workflow:

1. **📥 Load:**
   - Ingests the CCHS dataset (`health_dataset.csv`) alongside its metadata (`Data_dictionary.txt`).
   - Runs automatic quality checks to advise on missingness and data types.
   
2. **🧹 Clean:**
   - Provides a "Method-Aware Cleaning" station.
   - Automatically sanitizes invalid or missing codes across various features (e.g., changing 'Not stated' codes like 97, 98, 99 to `NaN` based on the data dictionary).
   - Allows users to restore previous versions of the dataset if needed.
   
3. **📊 Exploratory Data Analysis (EDA):**
   - **Univariate Analysis:** Distinguishes between numerical and categorical data to plot rich distributions, utilizing metadata to decode labels into human-readable formats.
   - **Bivariate Analysis:** Analyzes top correlations between features and a selected target variable.
   
4. **🔬 Experimental Lab (Modeling):**
   - Configure machine learning experiments (Target, Features, Ordinal vs Nominal categorical variables).
   - Contains safety checks to warn against potential data leakage.
   - Runs multiple regression algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Baseline).
   - Validates models with multiple random seeds to check stability.
   - Performs ablation (dependency quantification) to test feature sensitivity.
   - Exports the champion `.pkl` model.
   
5. **📝 Decision Report:**
   - Generates an executive summary of the experiment's readiness for deployment.
   - Evaluates risks like high variance, dependencies, and leakage.
   - Allows exporting the detailed decision report as Markdown.

## 📂 Project Structure

- `app.py`: The main Streamlit application script containing the UI and analytical logic.
- `health_dataset.csv`: The primary dataset containing modified CCHS data. *(Note: Must be present in the root directory)*
- `Data_dictionary.txt`: Text file containing descriptions and metadata for dataset columns.
- `Health_Dataset_Decoding_and_mapping_dictionary.docx`: Comprehensive guide for mapping and decoding variables.
- `ok.ipynb`: A supplementary Jupyter Notebook for experimental data analysis.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/savinaysingh7/Canadian-Community-Health-Survey-Predictive-Analytics.git
   cd Canadian-Community-Health-Survey-Predictive-Analytics
   ```
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scikit-learn scipy
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## ⚠️ Notes on Dataset Metadata

The project rigorously depends on the CCHS public-use documentation for handling categorical mappings, ordinal encodings, and invalid codes. A robust understanding of these variables can be found in `Data_dictionary.txt` and the provided decoding document.
