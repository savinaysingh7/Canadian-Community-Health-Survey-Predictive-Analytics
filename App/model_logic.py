import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

def validate_risk_input(data):
    """Validate user inputs for health risk prediction"""
    errors = []
    # Age is mapped 1-5
    if data.get('Age') not in [1, 2, 3, 4, 5]:
        errors.append("Invalid Age Group")
    # Physical activity range
    if data.get('Physical_vigorous_act_time', -1) < 0 or data.get('Physical_vigorous_act_time', 9999) > 500:
        errors.append("Physical activity out of range")
    # Smoking range
    if data.get('Smoked', -1) < 1 or data.get('Smoked', 9999) > 80:
        errors.append("Smoking magnitude out of range")
    return errors

def prepare_risk_input(data, template_df, explainer_model):
    """Prepare and feature engineer input data for the health risk model"""
    input_df = template_df.copy().reset_index(drop=True)
    
    # Extract preprocessor from explainer_model (which is the XGBoost pipeline)
    if hasattr(explainer_model, 'named_steps') and 'preprocessor' in explainer_model.named_steps:
        preprocessor = explainer_model.named_steps['preprocessor']
    else:
        raise ValueError("explainer_model must be a scikit-learn Pipeline with a 'preprocessor' step.")
    
    # Map basic inputs
    for col, val in data.items():
        if col in input_df.columns:
            if template_df[col].dtype == 'float64':
                input_df.loc[0, col] = float(val)
            else: 
                input_df.loc[0, col] = val

    # The preprocessor expects the original columns defined by template_df
    input_for_preprocessing = input_df[template_df.columns]

    # Transform the data using the extracted preprocessor
    input_df_preprocessed = preprocessor.transform(input_for_preprocessing)
    
    # Get feature names after preprocessing
    preprocessor_feature_names = preprocessor.get_feature_names_out(template_df.columns)
    
    return pd.DataFrame(input_df_preprocessed, columns=preprocessor_feature_names)

def prepare_wellbeing_input(data, feature_columns):
    """Prepare input data for the wellbeing (life satisfaction) model"""
    # Create a DataFrame with a single row from the data dictionary
    # Ensure all required features are present, defaulting to 0.0 if missing
    input_data = {}
    for col in feature_columns:
        input_data[col] = float(data.get(col, 0.0))
    
    return pd.DataFrame([input_data], columns=feature_columns)
