import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import uuid
import joblib
import tempfile
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr

# --------------------------------------------------
# PERSISTENCE HELPERS
# --------------------------------------------------
def save_experiments(exp_list: List['Experiment'], path="experiments.json"):
    try:
        with open(path, "w") as f:
            json.dump([e.__dict__ for e in exp_list], f, indent=4, default=str)
    except Exception as e:
        st.error(f"Failed to save experiments: {e}")

def load_experiments(path="experiments.json") -> List['Experiment']:
    try:
        with open(path, "r") as f:
            exp_dicts = json.load(f)
        exp_list = [Experiment(**d) for d in exp_dicts]
        return exp_list
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")
        return []

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Health Data Science Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {border-bottom: 2px solid #f0f2f6; padding-bottom: 1rem;}
    .stMetric {background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #eee;}
    .stAlert {padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# METADATA (from CCHS Data Dictionary)
# --------------------------------------------------
INVALID_CODES = {
    "Life_satisfaction": [97, 98, 99],
    "Worked_job_business": [6, 7, 8, 9],
    "Edu_level": [9],
    "Gen_health_state": [6, 9],
    "Mental_health_state": [7, 8, 9],
    "Stress_level": [7, 8],
    "Work_stress": [6, 7, 8, 9],
    "Sense_belonging": [7, 8, 9],
    "Weight_state": [6, 7, 8, 9],
    "Sleep_apnea": [7, 8],
    "High_BP": [7, 8],
    "High_cholestrol": [6, 7, 8],
    "Diabetic": [6, 7, 8],
    "Fatigue_syndrome": [7, 8],
    "Mood_disorder": [7, 8],
    "Anxiety_disorder": [7, 8],
    "Total_active_time": [9996, 9997, 9998, 9999],
    "Total_physical_act_time": [9996, 9998, 9999],
    "Work_hours": [96, 99],
}

ORDINAL_ORDER = {
    "Gen_health_state": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "Mental_health_state": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "Stress_level": [
        "Not at all stressful",
        "Not very stressful",
        "Somewhat stressful",
        "Quite a bit stressful",
        "Extremely stressful"
    ],
    "Work_stress": [
        "Not at all stressful",
        "Not very stressful",
        "Somewhat stressful",
        "Quite a bit stressful",
        "Extremely stressful",
        "Not applicable"
    ],
    "Sense_belonging": [
        "Very strong",
        "Somewhat strong",
        "Somewhat weak",
        "Very weak"
    ],
}

@dataclass
class Experiment:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    model_name: str = ""
    target: str = ""
    features: List[str] = field(default_factory=list)
    r2: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

# --------------------------------------------------
# UI / SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("Settings")
    data_path = st.text_input("Dataset Path", "health_dataset.csv")
    test_size = st.slider("Test Split Size", 0.1, 0.5, 0.2)
    random_seed = st.number_input("Random Seed", value=42)
    
    st.divider()
    st.info("This playground allows for rapid experimentation with CCHS data.")

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
st.title("🧠 Health Data Science Playground")

try:
    df_raw = pd.read_csv(data_path)
    st.success(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 1. CLEANING STEP
with st.expander("🛠️ Data Preparation", expanded=False):
    st.write("Applying CCHS-specific cleaning logic (Replacing invalid codes with NaN)")
    df_cleaned = df_raw.copy()
    for col, codes in INVALID_CODES.items():
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace(codes, np.nan)
    
    st.write(f"Rows with target (Life_satisfaction) missing: {df_cleaned['Life_satisfaction'].isna().sum()}")
    df_cleaned = df_cleaned.dropna(subset=['Life_satisfaction'])
    st.write(f"Final usable rows: {df_cleaned.shape[0]}")

# 2. EDA
tab_eda, tab_model, tab_history = st.tabs(["📊 Exploratory Analysis", "🤖 Model Training", "📜 Experiment History"])

with tab_eda:
    col1, col2 = st.columns(2)
    with col1:
        target_col = "Life_satisfaction"
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_cleaned[target_col], kde=True, bins=11, ax=ax)
        ax.set_title(f"Distribution of {target_col}")
        st.pyplot(fig)
    
    with col2:
        feature_for_corr = st.selectbox("Compare with Target", [c for c in df_cleaned.columns if c != target_col])
        if df_cleaned[feature_for_corr].dtype in ['int64', 'float64']:
            # Calculate spearman correlation
            temp_df = df_cleaned[[target_col, feature_for_corr]].dropna()
            corr, p = spearmanr(temp_df[target_col], temp_df[feature_for_corr])
            st.metric("Spearman Correlation", f"{corr:.3f}")
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df_cleaned[feature_for_corr], y=df_cleaned[target_col], ax=ax2)
            st.pyplot(fig2)

with tab_model:
    st.subheader("Pipeline Configuration")
    
    # Feature Selection
    all_features = [c for c in df_cleaned.columns if c != target_col]
    selected_features = st.multiselect("Select Features", all_features, 
                                     default=["Age", "Gender", "Gen_health_state", "Mental_health_state", "Stress_level", "Total_income"])
    
    if not selected_features:
        st.warning("Select at least one feature to continue.")
    else:
        # Automatic Type Detection
        numeric_features = [f for f in selected_features if df_cleaned[f].dtype in ['float64', 'int64'] and f not in ORDINAL_ORDER]
        ordinal_features = [f for f in selected_features if f in ORDINAL_ORDER]
        nominal_features = [f for f in selected_features if df_cleaned[f].dtype == 'object' and f not in ORDINAL_ORDER]
        
        col_type1, col_type2, col_type3 = st.columns(3)
        col_type1.write(f"Numeric: {len(numeric_features)}")
        col_type2.write(f"Ordinal: {len(ordinal_features)}")
        col_type3.write(f"Nominal: {len(nominal_features)}")
        
        st.divider()
        
        # Algorithm Selection
        algo_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression", "Decision Tree"])
        
        # Hyperparams
        params = {}
        if algo_choice in ["Random Forest", "Gradient Boosting"]:
            params['n_estimators'] = st.slider("n_estimators", 50, 500, 100)
            params['max_depth'] = st.slider("max_depth", 2, 20, 5)
        
        if st.button("🚀 Train Model"):
            with st.spinner("Building pipeline and training..."):
                # Data Split
                X = df_cleaned[selected_features]
                y = df_cleaned[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
                
                # Preprocessing
                transformers = []
                if numeric_features:
                    transformers.append(('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numeric_features))
                
                if nominal_features:
                    transformers.append(('nom', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), nominal_features))
                    
                if ordinal_features:
                    # Specific handling for ordinals
                    for feat in ordinal_features:
                        transformers.append((f'ord_{feat}', Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('ordinal', OrdinalEncoder(categories=[ORDINAL_ORDER[feat]]))
                        ]), [feat]))

                preprocessor = ColumnTransformer(transformers=transformers)
                
                # Model selection
                if algo_choice == "Random Forest":
                    model = RandomForestRegressor(**params, random_state=random_seed)
                elif algo_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(**params, random_state=random_seed)
                elif algo_choice == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = DecisionTreeRegressor(max_depth=params.get('max_depth', 5))
                
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Fit
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                duration = time.time() - start_time
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Results UI
                st.success(f"Training finished in {duration:.2f}s")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("R² Score", f"{r2:.4f}")
                res_col2.metric("MAE", f"{mae:.4f}")
                res_col3.metric("RMSE", f"{rmse:.4f}")
                
                # Feature Importance
                st.subheader("Explainability")
                if algo_choice in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
                    # Global importance
                    importances = pipeline.named_steps['model'].feature_importances_
                    # Get feature names from transformers
                    # This is complex in sklearn, but let's approximate or use permutation
                    r = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=random_seed)
                    
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': r.importances_mean
                    }).sort_values('importance', ascending=False)
                    
                    fig_imp, ax_imp = plt.subplots()
                    sns.barplot(x='importance', y='feature', data=importance_df, ax=ax_imp)
                    ax_imp.set_title("Permutation Importance (Test Set)")
                    st.pyplot(fig_imp)

                # Save Experiment
                new_exp = Experiment(
                    model_name=algo_choice,
                    target=target_col,
                    features=selected_features,
                    r2=r2, mae=mae, rmse=rmse,
                    params=params
                )
                current_exps = load_experiments()
                current_exps.append(new_exp)
                save_experiments(current_exps)
                
                # Export model
                st.divider()
                st.subheader("Model Export")
                fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(pipeline, fp.name)
                with open(fp.name, "rb") as f:
                    st.download_button(
                        "📦 Download Champion Model (.pkl)",
                        f,
                        file_name=f"model_{algo_choice.replace(' ', '_')}_{new_exp.id}.pkl"
                    )

with tab_history:
    st.subheader("Previous Runs")
    exps = load_experiments()
    if not exps:
        st.info("No experiments recorded yet.")
    else:
        history_df = pd.DataFrame([e.__dict__ for e in exps])
        st.dataframe(history_df.sort_values('r2', ascending=False), use_container_width=True)
        if st.button("Clear History"):
            save_experiments([])
            st.rerun()

st.divider()
st.caption("CCHS Predictive Analytics Platform | v1.0 • Built with Streamlit & Scikit-learn")
