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
        "A bit stressful",
        "Quite a bit stressful",
        "Extremely stressful",
    ],
    "Sense_belonging": ["Very strong", "Somewhat strong", "Somewhat weak", "Very weak"],
    "Weight_state": ["Underweight", "Just about right", "Overweight"],
    "Fruit_veg_con": ["Less than 5 times/day", "5-10 times/day", "More than 10 times/day"],
    "working_status": ["Full-time", "Part-time"],
    "Total_income": [
        "No income or less than $20,000",
        "$20,000 to $39,999",
        "$40,000 to $59,999",
        "$60,000 to $79,999",
        "$80,000 or more",
    ],
}

# Note: These are for visualization only. They are applied AFTER cleaning.
VALUE_MAPS = {
    "Gender": {1: "Male", 2: "Female"},
    "Marital_status": {1: "Married/Common-law", 2: "Widowed/Divorced/Separated/Single"},
    "Household": {1: "Lives alone", 2: "2 or more people"},
    "Worked_job_business": {1: "Yes", 2: "No"},
    "Edu_level": {1: "< Secondary", 2: "Secondary grad", 3: "Post-secondary+"},
    "Gen_health_state": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "Mental_health_state": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "Stress_level": {1: "Not at all", 2: "Not very", 3: "A bit", 4: "Quite a bit", 5: "Extremely"},
    "Sense_belonging": {1: "Very strong", 2: "Somewhat strong", 3: "Somewhat weak", 4: "Very weak"},
    "Diabetic": {1: "Yes", 2: "No"},
    "High_BP": {1: "Yes", 2: "No"},
    "Mood_disorder": {1: "Yes", 2: "No"},
    "Anxiety_disorder": {1: "Yes", 2: "No"},
    "Total_income": {
        1: "< $20k",
        2: "$20k–$40k",
        3: "$40k–$60k",
        4: "$60k–$80k",
        5: "$80k+",
    },
}

# --------------------------------------------------
# DATA STRUCTURES & LOGGING
# --------------------------------------------------
@dataclass
class Experiment:
    id: str
    timestamp: str
    target: str
    features: List[str]
    ordinal_feats: List[str]
    nominal_feats: List[str]
    models_run: List[str]
    seeds: List[int]
    best_model_name: str
    metrics: Dict[str, float]
    sensitivity_delta: float = 0.0
    sensitivity_feature: str = None
    leakage_warnings: List[str] = field(default_factory=list)
    stability_std: float = 0.0
    feature_roles: Dict[str, str] = field(default_factory=dict)
    

# --------------------------------------------------
# UTILITY CLASSES
# --------------------------------------------------
class DataUtils:
    @staticmethod
    @st.cache_data
    def load_data():
        return pd.read_csv("health_dataset.csv")

    @staticmethod
    @st.cache_data
    def load_data_dictionary():
        try:
            with open("Data_dictionary.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            data_dict = {}
            lines = content.strip().split('\n')
            in_table = False
            for line in lines:
                if "|---" in line:
                    in_table = True
                    continue
                if not in_table:
                    continue
                
                parts = line.split('|')
                if len(parts) > 2:
                    col_name = parts[1].strip()
                    description = parts[2].strip()
                    if col_name:
                        data_dict[col_name] = description
            return data_dict
        except FileNotFoundError:
            st.warning("Could not find Data_dictionary.txt. Descriptions will not be available.")
            return {}

    @staticmethod
    def check_quality(df):
        issues = []
        missing = df.isna().mean()
        high_missing_cols = missing[missing > 0.4].index.tolist()
        if high_missing_cols:
            issues.append(f"🔴 **High Missingness (>40%)**: {', '.join(high_missing_cols)}")

        num_cols = df.select_dtypes(include=np.number).columns
        for c in num_cols:
            if df[c].nunique() < 10 and df[c].nunique() > 1:
                issues.append(f"🟠 **Numeric-as-Categorical**: '{c}' has few unique values ({df[c].nunique()})")

        obj_cols = df.select_dtypes(include='object').columns
        for c in obj_cols:
            if df[c].str.isnumeric().mean() > 0.8:
                issues.append(f"🟡 **Potential Mixed Type**: '{c}' looks numeric but is object.")
        return issues

    @staticmethod
    def check_leakage(df, target, features):
        warnings = []
        if pd.api.types.is_numeric_dtype(df[target]):
            corrs = df[features].select_dtypes(include=np.number).corrwith(df[target]).abs()
            leakers = corrs[corrs > 0.95].index.tolist()
            if leakers:
                warnings.append(f"🚨 **Leakage Risk (>0.95 corr)**: {leakers}")
        
        for f in features:
            if df[f].nunique() == len(df):
                warnings.append(f"⚠️ **ID Column**: '{f}' is unique per row.")
        return warnings

    @staticmethod
    def validate_ordinal(df, feature, target):
        df_temp = df[[feature, target]].dropna()
        if pd.api.types.is_numeric_dtype(df_temp[feature]):
            corr, _ = spearmanr(df_temp[feature], df_temp[target])
            if abs(corr) < 0.1:
                return f"⚠️ '{feature}' marked Ordinal but has weak monotonic link (Spearman={corr:.2f})."
        return None

class ModelUtils:
    @staticmethod
    def get_regressors():
        return {
            "Baseline (Mean)": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }

    @staticmethod
    def train_and_evaluate(X, y, preprocessor, model_name, model_obj, seeds=[42]):
        scores = {'mae': [], 'rmse': [], 'r2': []}
        
        first_seed_artifacts = {}

        for i, seed in enumerate(seeds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            pipe = Pipeline([('prep', preprocessor), ('model', model_obj)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            
            scores['mae'].append(mean_absolute_error(y_test, preds))
            scores['rmse'].append(mean_squared_error(y_test, preds, squared=False))
            scores['r2'].append(r2_score(y_test, preds))

            if i == 0:
                first_seed_artifacts = {
                    'obj': pipe,
                    'X_test': X_test,
                    'y_test': y_test,
                    'residuals': y_test - preds,
                    'preds': preds
                }

        final_return = {
            'mae': np.mean(scores['mae']),
            'rmse': np.mean(scores['rmse']),
            'r2': np.mean(scores['r2']),
            'r2_std': np.std(scores['r2']),
        }
        final_return.update(first_seed_artifacts)
        return final_return

    @staticmethod
    def analyze_errors(metrics, target_std):
        insights = []
        if metrics['rmse'] > 1.5 * metrics['mae']:
            insights.append("🔴 RMSE ≫ MAE: Model is sensitive to outliers.")
        if metrics['r2'] > 0.8 and metrics['mae'] > 0.2 * target_std:
            insights.append("🟡 High Correlation but significant Absolute Error.")
        return insights

class ReportUtils:
    @staticmethod
    def generate_markdown(experiment: Experiment, logs: List[str]):
        lines = [f"# Data Science Decision Report", f"**Experiment ID:** {experiment.id}", f"**Date:** {experiment.timestamp}"]
        
        # 1. Executive Summary
        lines.append("\n## 1. Executive Summary")
        lines.append(f"- **Objective:** Predict `{experiment.target}`")
        lines.append(f"- **Champion Model:** {experiment.best_model_name}")
        lines.append(f"- **Key Metrics:** R²={experiment.metrics['r2']:.3f}, MAE={experiment.metrics['mae']:.3f}")
        
        # 2. Decision Verdict
        lines.append("\n## 2. Decision Readiness Verdict")
        risks = []
        if experiment.leakage_warnings:
            risks.append(f"🔴 **CRITICAL:** Potential leakage detected ({', '.join(experiment.leakage_warnings)}). Do not deploy.")
        if experiment.stability_std > 0.05:
            risks.append(f"🟠 **Unstable:** Model variance is high (Std={experiment.stability_std:.3f}). Results may not generalize.")
        if experiment.sensitivity_delta > 0.1:
            risks.append(f"🟡 **High Dependency:** Removing '{experiment.sensitivity_feature}' drops R² by {experiment.sensitivity_delta:.3f}. Model relies heavily on one feature.")
            
        if not risks:
            lines.append("✅ **Ready:** Model passes all stability, leakage, and dependency checks.")
        else:
            for r in risks: lines.append(r)
            
        # 3. Methodology
        lines.append("\n## 3. Methodology")
        lines.append(f"- **Features ({len(experiment.features)}):** {', '.join(experiment.features)}")
        if experiment.ordinal_feats:
            lines.append(f"- **Ordinal Encoding:** Ground-truth order used for: {', '.join(experiment.ordinal_feats)}")
        
        lines.append("\n## 4. Data Source Notes")
        lines.append("All variables and encodings follow the official Canadian Community Health Survey (CCHS) public-use documentation.")

        lines.append("\n## 5. Project Activity Log")
        for log in logs[-10:]:
            lines.append(f"- {log}")
            
        return "\n".join(lines)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if 'experiments' not in st.session_state:
    st.session_state.experiments = load_experiments()

DEFAULT_STATE = {
    "df_raw": None, "df_active": None, "history": [], "versions": {},
    "project_log": [], "data_dict": {},
    # For Re-run logic
    "set_target": None, "set_features": [], "set_ordinal": [],
    "set_models_to_run": None, "set_run_stability": True
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

def log_event(event):
    ts = time.strftime("%H:%M:%S")
    st.session_state.project_log.append(f"[{ts}] {event}")

# --------------------------------------------------
# MAIN APP UI
# --------------------------------------------------
st.sidebar.title("🧠 Data Playground")
st.sidebar.caption("v5.1 • Scientific Rigor")
step = st.sidebar.radio("Workflow", ["📥 Load", "🧹 Clean", "📊 EDA", "🔬 Lab (Model)", "📝 Report"])

if step == "📥 Load":
    st.title("Dataset Overview")
    if st.session_state.df_raw is None:
        st.session_state.df_raw = DataUtils.load_data()
        st.session_state.df_active = st.session_state.df_raw.copy()
        st.session_state.versions["Raw"] = st.session_state.df_raw.copy()
        st.session_state.data_dict = DataUtils.load_data_dictionary()
        log_event("Loaded dataset and data dictionary")
        st.rerun()
        
    df = st.session_state.df_active
    st.dataframe(df.head(), use_container_width=True)
    issues = DataUtils.check_quality(df)
    if issues:
        with st.expander("💡 Quality Advisor", expanded=True):
            for i in issues: st.markdown(i)

elif step == "🧹 Clean":
    st.title("Cleaning Station")
    df = st.session_state.df_active
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Method-Aware Cleaning")
        if st.button("✨ Auto-Clean (Apply Dictionary Rules)"):
            before_na = df.isna().sum()
            df_cleaned = df.copy()
            cleaned_cols = []
            for col, codes in INVALID_CODES.items():
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].replace(codes, np.nan)
                    cleaned_cols.append(col)
            
            after_na = df_cleaned.isna().sum()
            diff = (after_na - before_na).loc[after_na - before_na > 0]
            rows_affected = (df.ne(df_cleaned)).any(axis=1).sum()

            st.session_state.df_active = df_cleaned
            st.session_state.versions["Method-Cleaned"] = df_cleaned.copy()
            log_event(f"Applied dictionary-based cleaning to {len(cleaned_cols)} columns, affecting {rows_affected} rows.")
            st.success(f"Cleaned {len(cleaned_cols)} columns based on data dictionary rules.")

            if not diff.empty:
                st.markdown("##### Cleaning Audit Report")
                st.table(diff.rename("New missing values"))
                st.info(f"**{rows_affected}** rows were affected by this cleaning operation.")
            else:
                st.info("No new missing values were introduced by this operation.")

    with col2:
        st.subheader("History")
        version_keys = list(st.session_state.versions.keys())
        if version_keys:
            selected_version = st.radio("Select a version to restore", version_keys, index=len(version_keys)-1)
            if st.button("↩️ Restore Version"):
                st.session_state.df_active = st.session_state.versions[selected_version]
                log_event(f"Restored to version: {selected_version}")
                st.rerun()

elif step == "📊 EDA":
    st.title("📊 Exploratory Analysis")
    df = st.session_state.df_active

    st.subheader("Univariate Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        eda_col = st.selectbox("Select a column to analyze", df.columns, key="eda_col_select")
    with col2:
        decode = st.checkbox("Decode Labels", value=True, help="Show human-readable labels for categorical data in plots.")

    if eda_col:
        if eda_col in st.session_state.data_dict:
            st.info(f"**Description**: {st.session_state.data_dict[eda_col]}")

        # --- METADATA-AWARE PLOTTING ---
        is_categorical = eda_col in VALUE_MAPS
        
        if is_categorical and decode:
            st.markdown("Distribution of categories (decoded)")
            plot_series = df[eda_col].map(VALUE_MAPS[eda_col]).dropna()
            fig, ax = plt.subplots()
            sns.countplot(y=plot_series, ax=ax, order=plot_series.value_counts().index, palette="viridis")
            ax.set_title(f"Distribution of {eda_col}")
            ax.set_xlabel("Count")
            ax.set_ylabel("Category")
            plt.tight_layout()
            st.pyplot(fig)
            
        elif pd.api.types.is_numeric_dtype(df[eda_col]):
            st.markdown("Distribution of numeric values")
            fig, ax = plt.subplots()
            sns.histplot(df[eda_col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {eda_col}")
            ax.set_xlabel("Value")
            st.pyplot(fig)
        else: # Handle non-decoded categoricals or high-cardinality objects
            st.markdown("Distribution of values")
            st.dataframe(df[eda_col].value_counts())

    st.divider()

    st.subheader("Bivariate Analysis (Correlation)")
    num_cols = df.select_dtypes(include=np.number).columns
    with st.expander("Bridge to Modeling: Target Correlation", expanded=False):
        target_corr = st.selectbox("Select a potential numeric target", num_cols, key="corr_target_select")
        if target_corr:
            if target_corr in st.session_state.data_dict:
                st.info(f"**Description for `{target_corr}`**: {st.session_state.data_dict[target_corr]}")
            
            # Calculate correlations
            corrs = df[num_cols].corrwith(df[target_corr]).sort_values(ascending=False).drop(target_corr)
            
            # Display chart
            st.markdown(f"**Top 10 Features Correlated with `{target_corr}`**")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10_corrs = corrs.head(10)
            sns.barplot(x=top_10_corrs.values, y=top_10_corrs.index, ax=ax, palette="vlag")
            ax.set_xlabel("Correlation Coefficient")
            st.pyplot(fig)

elif step == "🔬 Lab (Model)":
    st.title("🔬 Experimental Lab")
    df = st.session_state.df_active.copy().dropna()
    
    # Warn if dataset is very small after dropping NaNs
    if df.shape[0] < 200:
        st.warning(f"️⚠️ Only {df.shape[0]} rows available after removing missing values. Model may be unstable.")
    
    if df.empty:
        st.error("Empty dataset. Please clean first.")
        st.stop()
        
    # --- REPRODUCIBILITY LOADER ---
    if st.session_state.experiments:
        with st.expander("📂 Load Previous Experiment Config"):
            sorted_exps = sorted(st.session_state.experiments, key=lambda e: e.timestamp, reverse=True)
            prev_exp_opts = [f"{e.timestamp} - {e.id} ({e.target})" for e in sorted_exps]
            
            load_idx = st.selectbox("Select Experiment to Reload", range(len(prev_exp_opts)), format_func=lambda x: prev_exp_opts[x])
            
            if st.button("Load Config"):
                loaded = sorted_exps[load_idx]
                st.session_state.set_target = loaded.target
                st.session_state.set_features = loaded.features
                st.session_state.set_ordinal = loaded.ordinal_feats
                st.session_state.set_models_to_run = loaded.models_run
                st.session_state.set_run_stability = len(loaded.seeds) > 1
                log_event(f"Loaded config from experiment {loaded.id} for re-run.")
                st.rerun()

    # --- SETUP ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Setup")
        
        # Target
        target_default_idx = 0
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if st.session_state.set_target in num_cols:
            target_default_idx = num_cols.index(st.session_state.set_target)
        target = st.selectbox("Target Variable", num_cols, index=target_default_idx)
        if target and target in st.session_state.data_dict:
            st.caption(f"**{target}**: {st.session_state.data_dict[target]}")

        # Features
        available_feats = [c for c in df.columns if c != target]
        features = st.multiselect("Features", available_feats, default=st.session_state.set_features if st.session_state.set_features else available_feats[:3])
        if features:
            with st.expander("Feature Descriptions"):
                for f in features:
                    if f in st.session_state.data_dict:
                        st.markdown(f"- **{f}**: {st.session_state.data_dict[f]}")
                    else:
                        st.markdown(f"- **{f}**: No description available.")
        
        st.markdown("---")
        st.markdown("**Encoding Strategy**")

        # Identify categorical features based on metadata, not just dtype
        cat_feats = [f for f in features if df[f].dtype == 'object' or f in ORDINAL_ORDER or f in VALUE_MAPS]
        
        # Suggest ordinal features based on our ground-truth metadata
        suggested_ordinals = [f for f in cat_feats if f in ORDINAL_ORDER]
        default_ordinals = st.session_state.set_ordinal if st.session_state.get("set_ordinal") else suggested_ordinals
        
        ordinal_feats = st.multiselect("Ordinal Features (Ranked)", cat_feats, default=default_ordinals)
        nominal_feats = [f for f in cat_feats if f not in ordinal_feats]
        
        # Defensive Ordinal
        ordinal_ok = True
        for ord_f in ordinal_feats:
            warn = DataUtils.validate_ordinal(df, ord_f, target)
            if warn: 
                st.caption(warn)
                if not st.checkbox(f"Confirm '{ord_f}' is Ordinal?", key=f"confirm_{ord_f}"):
                    ordinal_ok = False

    with col2:
        st.subheader("2. Model Config")
        models_to_run = st.multiselect("Algorithms", ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"], default=st.session_state.get("set_models_to_run") or ["Linear Regression", "Random Forest"])
        run_stability = st.checkbox("Run Stability Check (3 seeds)", value=st.session_state.get("set_run_stability", True))
        
        leaks = DataUtils.check_leakage(df, target, features)
        if leaks:
            st.error("⚠️ Safety Warnings:")
            for l in leaks: st.text(l)

    btn_disabled = not features or not ordinal_ok
    if st.button("🚀 Run Experiment", type="primary", disabled=btn_disabled):
        exp_id = uuid.uuid4().hex[:8]
        X = df[features]
        y = df[target]
        num_feats = [f for f in features if f not in cat_feats]
        
        # Create the list of transformers for the pipeline
        transformers = [('num', StandardScaler(), num_feats)]
        if nominal_feats: 
            transformers.append(('nom', OneHotEncoder(handle_unknown='ignore'), nominal_feats))
        
        # --- Metadata-Aware Ordinal Encoding (Safe) ---
        if ordinal_feats:
            ordered_categories = []
            for f in ordinal_feats:
                if f in ORDINAL_ORDER and f in VALUE_MAPS:
                    reverse_map = {v: k for k, v in VALUE_MAPS[f].items()}
                    # Safely map codes, skipping any that might not exist in the reverse map
                    ordered_codes = [reverse_map.get(val) for val in ORDINAL_ORDER[f] if reverse_map.get(val) is not None]
                    
                    if ordered_codes and len(ordered_codes) == len(ORDINAL_ORDER[f]):
                        ordered_categories.append(ordered_codes)
                    else:
                        ordered_categories.append('auto') # Fallback if mapping is incomplete
                else:
                    ordered_categories.append('auto') # Fallback for user-defined ordinals
            
            transformers.append(('ord', OrdinalEncoder(categories=ordered_categories), ordinal_feats))
        
        preprocessor = ColumnTransformer(transformers)
        
        results = []
        regressors = ModelUtils.get_regressors()
        seeds = [42, 52, 62] if run_stability else [42]
        progress = st.progress(0)
        
        # MAIN LOOP
        for i, m_name in enumerate(["Baseline (Mean)"] + models_to_run):
            if m_name not in regressors: continue
            res = ModelUtils.train_and_evaluate(X, y, preprocessor, m_name, regressors[m_name], seeds)
            results.append({
                "Model": m_name,
                "R2": res['r2'], "MAE": res['mae'], "RMSE": res['rmse'], "Std": res['r2_std'],
                "Obj": res.get('obj'), "X_test": res.get('X_test'), "y_test": res.get('y_test'), 
                "Residuals": res.get('residuals'), "Predictions": res.get('preds')
            })
            progress.progress(int(((i+1)/(len(models_to_run)+1)) * 100))
        progress.empty()
        
        res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
        best_row = res_df.iloc[0]
        
        # DEPENDENCY QUANTIFICATION (Ablation)
        sens_delta = 0.0
        sens_feat = None
        if best_row["Model"] != "Baseline (Mean)" and best_row['Obj'] is not None:
            perm_res = permutation_importance(best_row['Obj'], best_row['X_test'], best_row['y_test'], n_repeats=5, random_state=42)
            top_idx = perm_res.importances_mean.argmax()
            sens_feat = features[top_idx]
            
            # Retrain best algorithm without top feature
            X_drop = X.drop(columns=[sens_feat])
            
            # Rebuild transformer dynamically
            feats_drop = [f for f in features if f != sens_feat]
            num_drop = [f for f in feats_drop if f not in cat_feats]
            nom_drop = [f for f in nominal_feats if f != sens_feat]
            ord_drop = [f for f in ordinal_feats if f != sens_feat]
            
            trans_drop = [('num', StandardScaler(), num_drop)]
            if nom_drop: trans_drop.append(('nom', OneHotEncoder(handle_unknown='ignore'), nom_drop))
            if ord_drop: trans_drop.append(('ord', OrdinalEncoder(), ord_drop))
            prep_drop = ColumnTransformer(trans_drop)
            
            # Train once with main seed
            res_drop = ModelUtils.train_and_evaluate(X_drop, y, prep_drop, best_row["Model"], regressors[best_row["Model"]], seeds=[42])
            sens_delta = best_row["R2"] - res_drop["r2"]

        # DISPLAY RESULTS
        st.divider()
        st.subheader(f"Experiment Results (ID: {exp_id})")
        
        c_res, c_chart = st.columns([1,1])
        with c_res:
            st.dataframe(res_df[['Model', 'R2', 'MAE', 'RMSE', 'Std']].style.highlight_max(subset=['R2'], color='#90ee90'))
            
            if sens_feat:
                st.info(f"🧩 **Dependency Check:** Removing '{sens_feat}' drops R² by **{sens_delta:.3f}**")

            # --- Model Download ---
            if best_row['Obj'] is not None:
                fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(best_row['Obj'], fp.name)
                with open(fp.name, "rb") as f:
                    st.download_button(
                        "📦 Download Champion Model (.pkl)", 
                        f.read(), 
                        file_name=f"model_{best_row['Model'].replace(' ', '_')}_{exp_id}.pkl"
                    )
                
        with c_chart:
            st.markdown("**Error Diagnostics (Best Model)**")
            if best_row['Residuals'] is not None and best_row['Predictions'] is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Residuals Distribution
                sns.histplot(best_row['Residuals'], kde=True, ax=axes[0], color='skyblue')
                axes[0].set_title(f"Residuals Distribution ({best_row['Model']})")
                axes[0].set_xlabel("Residual")
                
                # Residuals vs. Predicted
                sns.scatterplot(x=best_row['Predictions'], y=best_row['Residuals'], ax=axes[1], alpha=0.5)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_title("Residuals vs. Predicted")
                axes[1].set_xlabel("Predicted Value")
                axes[1].set_ylabel("Residual")

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("Could not generate error plots (e.g., for Baseline model).")

        # LOGGING
        exp = Experiment(
            id=exp_id, timestamp=time.strftime("%Y-%m-%d %H:%M"),
            target=target, features=features, ordinal_feats=ordinal_feats, nominal_feats=nominal_feats,
            models_run=models_to_run, 
            seeds=seeds,
            best_model_name=best_row['Model'],
            metrics={'r2': best_row['R2'], 'mae': best_row['MAE']},
            sensitivity_delta=sens_delta, sensitivity_feature=sens_feat,
            leakage_warnings=leaks, stability_std=best_row['Std']
        )
        st.session_state.experiments.append(exp)
        save_experiments(st.session_state.experiments)
        log_event(f"Run Exp {exp_id}: Best={best_row['Model']}")

elif step == "📝 Report":
    st.title("📝 Decision Report")
    if not st.session_state.experiments:
        st.info("Run an experiment first in the '🔬 Lab (Model)' tab.")
        st.stop()
        
    exp_opts = [f"{e.timestamp} - {e.id} ({e.best_model_name})" for e in st.session_state.experiments]
    selected_idx = st.selectbox("Select Experiment to Report On", range(len(exp_opts)), format_func=lambda x: exp_opts[x], index=len(exp_opts)-1)
    experiment = st.session_state.experiments[selected_idx]

    # --- Feature Role Annotation ---
    with st.expander("Annotate Feature Roles for Report"):
        st.markdown("Classify each feature based on its business context. This will be included in the report.")
        roles = ['Descriptive', 'Actionable', 'Sensitive']
        for f in experiment.features:
            current_role = experiment.feature_roles.get(f, 'Descriptive')
            current_role_index = roles.index(current_role) if current_role in roles else 0
            
            new_role = st.selectbox(
                f"**{f}**",
                options=roles,
                index=current_role_index,
                key=f"role_{experiment.id}_{f}"
            )
            experiment.feature_roles[f] = new_role

    # --- Report Generation ---
    st.divider()
    st.subheader("Generated Report")
    report_md = ReportUtils.generate_markdown(experiment, st.session_state.project_log)
    
    report_md += "\n\n## 6. Decision Factors (Feature Roles)\n"
    if experiment.feature_roles:
        for f, role in experiment.feature_roles.items():
            report_md += f"- **{f}**: `{role}`\n"
    else:
        report_md += "No feature roles have been assigned for this experiment."
            
    st.markdown(report_md)
    st.download_button("📥 Download Report as Markdown", report_md, f"Report_{experiment.id}.md", "text/markdown")