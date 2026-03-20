import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import database_manager as db
import auth_manager as auth
from model_logic import validate_risk_input, prepare_risk_input, prepare_wellbeing_input
from datetime import datetime
import os

# --- Configuration ---
st.set_page_config(
    page_title="HealthAI Pro | Unified Clinical & Wellbeing Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize System ---
db.init_db()
auth.init_auth()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: 600; }
    .risk-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    .high-risk { background-color: #ffebee; border: 1px solid #ffcdd2; color: #c62828; }
    .low-risk { background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }
    .wellbeing-card { background-color: #e3f2fd; border: 1px solid #bbdefb; color: #0d47a1; padding: 20px; border-radius: 10px; text-align: center; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    # In consolidated structure, Models are in ../Models/
    models_path = os.path.join("..", "Models")
    try:
        # Clinical Models
        risk_model = joblib.load(os.path.join(models_path, 'best_health_ensemble.pkl'))
        risk_explainer = joblib.load(os.path.join(models_path, 'best_xgb_model.pkl'))
        risk_template = joblib.load(os.path.join(models_path, 'template_df.pkl'))
        with open(os.path.join(models_path, 'app_metadata.json'), 'r') as f:
            risk_meta = json.load(f)
        
        # Wellbeing Models
        wellbeing_artifact = joblib.load(os.path.join(models_path, 'artifact_best_model_with_meta.pkl'))
        wellbeing_model = wellbeing_artifact['model']
        wellbeing_features = wellbeing_artifact['features']
        
        return risk_model, risk_explainer, risk_template, risk_meta, wellbeing_model, wellbeing_features
    except Exception as e:
        st.error(f"⚠️ System Offline: Model artifacts missing or corrupted. Error: {e}")
        st.stop()

risk_model, risk_explainer, risk_template, risk_meta, wellbeing_model, wellbeing_features = load_artifacts()

# --- Auth Pages ---
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 HealthAI Pro Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = auth.check_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

# --- App Pages ---
def dashboard_page():
    st.title("📊 Clinical & Wellbeing Dashboard")
    assessments = db.get_all_assessments()
    wellbeing = db.get_all_wellbeing_assessments()
    patients = db.get_all_patients()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(patients))
    col2.metric("Risk Assessments", len(assessments))
    col3.metric("Wellbeing Checks", len(wellbeing))
    
    high_risk_count = sum(1 for a in assessments if a['prediction'] == 1)
    col4.metric("High Risk Patients", high_risk_count)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recent Clinical Risk Assessments")
        if assessments:
            df = pd.DataFrame(assessments)
            st.dataframe(df[['date', 'patient_id', 'risk_score', 'prediction']].tail(5), use_container_width=True)
        else:
            st.info("No risk assessments yet.")
            
    with c2:
        st.subheader("Recent Wellbeing Assessments")
        if wellbeing:
            df_w = pd.DataFrame(wellbeing)
            st.dataframe(df_w[['date', 'patient_id', 'wellbeing_score']].tail(5), use_container_width=True)
        else:
            st.info("No wellbeing assessments yet.")

def patients_page():
    st.title("👤 Patient Management")
    
    with st.expander("➕ Add New Patient"):
        with st.form("add_patient"):
            name = st.text_input("Full Name")
            age_group_int = st.selectbox("Age Group", options=[1, 2, 3, 4, 5], 
                                   format_func=lambda x: {1: '12-17', 2: '18-34', 3: '35-49', 4: '50-64', 5: '65+'}[x])
            gender_int = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
            if st.form_submit_button("Save Patient"):
                if name:
                    db.add_patient(name, age_group_int, gender_int)
                    st.success(f"Patient '{name}' added!")
                    st.rerun()
                else:
                    st.error("Please enter a name.")
                
    st.subheader("Patient Registry")
    patients = db.get_all_patients()
    if patients:
        df = pd.DataFrame(patients)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No patients registered yet.")

def clinical_assessment_page():
    st.title("🩺 Clinical Risk Assessment")
    
    patients = db.get_all_patients()
    if not patients:
        st.warning("Please add a patient first.")
        return

    patient_opts = {p['id']: f"{p['name']} (ID: {p['id']})" for p in patients}
    selected_pid = st.selectbox("Select Patient", options=list(patient_opts.keys()), format_func=lambda x: patient_opts[x])
    
    st.markdown("---")
    
    # Input Mappings
    age_map = {1: '12-17', 2: '18-34', 3: '35-49', 4: '50-64', 5: '65+'}
    bmi_map = {1: 'Normal/Underweight', 2: 'Overweight/Obese'}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Vitals")
        age = st.selectbox("Age Group", list(age_map.keys()), format_func=lambda x: age_map[x], index=3)
        bmi = st.selectbox("BMI Category", [1, 2], format_func=lambda x: bmi_map[x], index=1)
        bp = st.radio("High BP?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        chol = st.radio("High Cholesterol?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        st.subheader("Mental Health")
        gen_h = st.slider("General Health (1=Ex, 5=Poor)", 1, 5, 3)
        men_h = st.slider("Mental Health (1=Ex, 5=Poor)", 1, 5, 3)
        stress = st.slider("Stress Level (1=Low, 5=High)", 1, 5, 3)
        satisfaction = st.slider("Life Satisfaction (0-10)", 0, 10, 7)

    with col3:
        st.subheader("Lifestyle")
        smoke = st.slider("Smoking (Cigs/Day)", 0, 80, 0)
        activity = st.slider("Activity (Min/Week)", 0, 500, 150)
        fruit = st.selectbox("Fruit/Veg Consumption", [1, 2, 3], format_func=lambda x: ["Low", "Med", "High"][x-1])
        sleep_apnea = st.radio("Sleep Apnea?", [1, 0])

    if st.button("Predict Clinical Risk"):
        input_data = {
            'Age': age, 'BMI_18_above': bmi, 'High_BP': float(bp), 'High_cholestrol': float(chol),
            'Gen_health_state': gen_h, 'Mental_health_state': men_h, 'Stress_level': stress,
            'Life_satisfaction': float(satisfaction), 'Smoked': float(smoke),
            'Physical_vigorous_act_time': float(activity), 'Fruit_veg_con': fruit, 'Sleep_apnea': float(sleep_apnea),
            'Gender': 1, 'Work_stress': 6, 'Cardiovascular_con': 0.0, 'Mood_disorder': 0.0, 'Anxiety_disorder': 0.0
        }
        
        errors = validate_risk_input(input_data)
        if errors:
            st.error(f"Errors: {', '.join(errors)}")
            return

        input_df = prepare_risk_input(input_data, risk_template, risk_explainer)
        prob = risk_model.predict_proba(input_df)[0][1]
        is_high = prob >= risk_meta['optimal_threshold']
        
        db.log_assessment(selected_pid, input_data, prob, is_high)
        
        res_class = "high-risk" if is_high else "low-risk"
        st.markdown(f"<div class='risk-card {res_class}'><h2>{'HIGH RISK' if is_high else 'LOW RISK'}</h2><h1>{prob:.1%}</h1></div>", unsafe_allow_html=True)
        st.success("Clinical assessment saved.")

def wellbeing_assessment_page():
    st.title("🌟 Wellbeing & Life Satisfaction")
    
    patients = db.get_all_patients()
    if not patients:
        st.warning("Please add a patient first.")
        return

    patient_opts = {p['id']: f"{p['name']} (ID: {p['id']})" for p in patients}
    selected_pid = st.selectbox("Select Patient for Wellbeing Check", options=list(patient_opts.keys()), format_func=lambda x: patient_opts[x])
    
    st.markdown("---")
    st.subheader("Input Wellbeing Factors")
    
    user_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(wellbeing_features):
        with cols[i % 3]:
            # Human-readable labels would be better, but using feature names for now
            user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Life Satisfaction"):
        input_df = prepare_wellbeing_input(user_inputs, wellbeing_features)
        prediction = wellbeing_model.predict(input_df)[0]
        
        db.log_wellbeing_assessment(selected_pid, user_inputs, prediction)
        
        st.markdown(f"<div class='wellbeing-card'><h2>Predicted Life Satisfaction Score</h2><h1>{prediction:.2f} / 10</h1></div>", unsafe_allow_html=True)
        st.success("Wellbeing assessment saved.")

def analytics_page():
    st.title("📈 Population Health Analytics")
    
    assessments = db.get_all_assessments()
    wellbeing = db.get_all_wellbeing_assessments()
    
    if not assessments and not wellbeing:
        st.info("Insufficient data for analytics.")
        return

    tab1, tab2 = st.tabs(["Clinical Risk Trends", "Wellbeing Distribution"])
    
    with tab1:
        if assessments:
            df = pd.DataFrame(assessments)
            fig, ax = plt.subplots()
            sns.histplot(df['risk_score'], bins=20, kde=True, ax=ax, color='salmon')
            ax.set_title("Distribution of Clinical Risk Scores")
            st.pyplot(fig)
        else:
            st.write("No clinical data.")

    with tab2:
        if wellbeing:
            dfw = pd.DataFrame(wellbeing)
            fig, ax = plt.subplots()
            sns.boxplot(y=dfw['wellbeing_score'], ax=ax, color='skyblue')
            ax.set_title("Life Satisfaction Distribution")
            st.pyplot(fig)
        else:
            st.write("No wellbeing data.")

# --- Main App Logic ---
if not st.session_state.logged_in:
    login_page()
else:
    with st.sidebar:
        st.title("🩺 HealthAI Pro")
        st.write(f"Doctor: **{st.session_state.user['username']}**")
        st.markdown("---")
        menu = st.radio("Navigation", ["Dashboard", "Patients", "Clinical Risk", "Wellbeing Check", "Analytics"])
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    if menu == "Dashboard":
        dashboard_page()
    elif menu == "Patients":
        patients_page()
    elif menu == "Clinical Risk":
        clinical_assessment_page()
    elif menu == "Wellbeing Check":
        wellbeing_assessment_page()
    elif menu == "Analytics":
        analytics_page()
