import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path, sample_fraction=0.05, chunk_size=10000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    return pd.concat([chunk.sample(frac=sample_fraction) for chunk in chunks], ignore_index=True)

def clean_data_types(data):
    data['CLoan_to_value'] = pd.to_numeric(data['CLoan_to_value'], errors='coerce')
    data['OLoan_to_value'] = pd.to_numeric(data['OLoan_to_value'], errors='coerce')

    indicator_cols = ['Single_borrower', 'Number_of_units', 'DFlag']
    for col in indicator_cols:
        data[col] = data[col].astype(str)

    column_prefixes = ['is_Loan_purpose', 'is_First_time_', 'is_Occupancy_status_',
                       'is_Origination_', 'is_Property_type_', 'is_property_state_',
                       'MSA_', 'Seller_']
    for prefix in column_prefixes:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].apply(lambda x: x.astype('object'))

    return data

def preprocess_data(df):
    df.dropna(inplace=True)

    prefix = ('Loanref', 'MSA_', 'PostalCode_', 'Seller_', 'is_property_state_')
    df.drop(columns=[col for col in df.columns if col.startswith(prefix)], inplace=True)

    df_original = df.copy()

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    target = 'DFlag'
    df[target] = LabelEncoder().fit_transform(df[target])
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, df_original

def preprocess_for_prediction(input_data, feature_columns, scaler):
    missing_features = [col for col in feature_columns if col not in input_data.columns]
    for feature in missing_features:
        input_data[feature] = 0  # Default value for missing features

    input_data = input_data[feature_columns]
    #return scaler.transform(input_data)

def shap_analysis(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as e:
        print(f"Error during SHAP calculation: {e}")

    if X.shape[1] != len(feature_names):
        raise ValueError("Mismatch between SHAP feature names and dataset dimensions")

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)

    for feature in feature_names[:5]:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X, feature_names=feature_names, show=False)

def make_prediction(model, scaler, selected_features, df_cleaned):
    df_cleaned = df_cleaned[selected_features]
    st.write(selected_features)
    numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    inputs = {}
    for col in numerical_cols:
        inputs[col] = st.number_input(f"{col}", value=float(df_cleaned[col].mean()))

    for col in categorical_cols:
        unique_vals = df_cleaned[col].unique()
        inputs[col] = st.selectbox(f"{col}", options=unique_vals)

    input_df = pd.DataFrame([inputs])
    input_scaled = preprocess_for_prediction(input_df, selected_features, scaler)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        st.write(f"Predicted Class: {'Default' if prediction == 1 else 'No Default'}")

def visualize_train_data(data):
    st.subheader("Train Data Preview")
    st.markdown("### Sample Data")
    st.write(data.head())

def display_insights(data):
    st.subheader("Insights")
    st.markdown("### Basic Statistics")
    st.write(data.describe())

def display_visualizations(data):
    st.subheader("Visualizations")
    st.markdown("### Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

def main():
    st.set_page_config(layout='wide')
    st.title("Credit Risk Analysis Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Train Data Preview", "Insights", "Visualizations"])

    base_dir = os.path.dirname(__file__)
    data_folder = os.path.join(base_dir, "data")

    model = joblib.load(os.path.join(data_folder, "best_model_v2.pkl"))
    selected_features = joblib.load(os.path.join(data_folder, "selected_features.pkl"))

    file_path = st.sidebar.text_input("Enter Dataset Path:", os.path.join(data_folder, "red_train_12.csv"))

    if file_path:
        data = load_data(file_path)
        data = clean_data_types(data)

        X, y, scaler, df_cleaned = preprocess_data(data)

        with tab1:
            st.subheader("Make Prediction")
            make_prediction(model, scaler, selected_features, df_cleaned)

        with tab2:
            visualize_train_data(data)

        with tab3:
            display_insights(data)

        with tab4:
            display_visualizations(data)

if __name__ == "__main__":
    main()
