import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Page configuration
st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

# Custom Styling for Dark Mode


base_dir = os.path.dirname(__file__)

data_folder = os.path.join(base_dir, "data")

# Load model and preprocessing artifacts
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(data_folder, "best_model_v1.pkl"))
    scaler = joblib.load(os.path.join(data_folder, "scaler_v1.pkl"))
    selected_features = joblib.load(os.path.join(data_folder, "selected_features_v1.pkl"))
    return model, scaler, selected_features

model, scaler, selected_features = load_model()

df_train = pd.read_csv(os.path.join(data_folder, "preprocessed_v1.csv"))
label_encoders = {col: LabelEncoder().fit(df_train[col]) for col in df_train.select_dtypes(include=['object']).columns}

# Sidebar
st.sidebar.header("📊 Churn Prediction Dashboard")
st.sidebar.subheader("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Tabs
tabs = st.tabs(["📁 Train Data", "🔮 Manual Prediction", "📂 Batch Prediction", "📊 Insights & Visuals"])

# Train Data Preview
with tabs[0]:
    st.subheader("📁 Training Data Sample")

    st.info(
    "🔍 **Churn Classification:** Customers whose last order was **over 6 months (180 days) ago** are marked as **Churned** 🛑, while recent buyers remain **Active** ✅.")

    st.write(df_train.head())

    fig1 = px.histogram(df_train, x='Recency', title="Recency Distribution: Days Since Last Purchase", nbins=30, color_discrete_sequence=['#4CAF50'])
    fig2 = px.histogram(df_train, x='Frequency', title="Frequency Distribution: No. of Transactions Per Customer", nbins=30, color_discrete_sequence=['#FF9800'])
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# Manual Churn Prediction
with tabs[1]:
    st.subheader("🔮 Manual Prediction")
    user_input = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(selected_features):
        if df_train[feature].dtype == 'object':
            options = df_train[feature].dropna().unique().tolist()
            user_input[feature] = col1.selectbox(f"{feature}", options) if i % 2 == 0 else col2.selectbox(f"{feature}", options)
        else:
            user_input[feature] = col1.number_input(f"{feature}", value=df_train[feature].median()) if i % 2 == 0 else col2.number_input(f"{feature}", value=df_train[feature].median())
    
    if st.button("Predict Churn"):
        user_df = pd.DataFrame([user_input])
        for col, encoder in label_encoders.items():
            if col in user_df.columns:
                user_df[col] = encoder.transform(user_df[col])
        user_scaled = scaler.transform(user_df)
        churn_prob = model.predict_proba(user_scaled)[:, 1][0] 
        churn_pred = model.predict(user_scaled)[0]

        st.success(f"Churn Probability: {churn_prob:.2f}")
        st.success(f"Churn Prediction: {'Yes' if churn_pred == 1 else 'No'}")

# Batch Prediction
with tabs[2]:
    st.subheader("📂 Batch Prediction")
    st.info("upload your csv(check sample below)")
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
    else:
        df_batch = df_train[selected_features].copy()
    
    st.write(df_batch.head())
    df_selected = df_batch[selected_features].copy()
    
    for col, encoder in label_encoders.items():
        if col in df_selected.columns:
            df_selected[col] = encoder.transform(df_selected[col])
    df_scaled = scaler.transform(df_selected)

    churn_probs = model.predict_proba(df_scaled)[:, 1]
    df_batch['Churn Probability'] = churn_probs
    df_batch['Churn Prediction'] = np.where(churn_probs > 0.5, 1, 0)
    df_batch['Churn Prediction[Normalized]'] = np.where(churn_probs > 0.5, 'Yes', 'No')

    st.write("##### Churn Predictions")
    st.dataframe(df_batch[['Churn Probability', 'Churn Prediction', 'Churn Prediction[Normalized]']].head())

    fig3 = px.histogram(df_batch, x='Churn Probability', nbins=30, title='Churn Probability Distribution', color_discrete_sequence=['#E91E63'])
    st.plotly_chart(fig3, use_container_width=True)

# Insights & Visualizations
with tabs[3]:
    st.subheader("📊 Insights & Visuals")
    df = df_batch if uploaded_file is None else uploaded_file

    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        churn_rate = df.groupby(df['OrderDate'].dt.to_period("M"))['Churn Prediction'].mean().reset_index()
        churn_rate['OrderDate'] = churn_rate['OrderDate'].astype(str)
        fig4 = px.line(churn_rate, x='OrderDate', y='Churn Prediction', title="Churn Rate Over Time", color_discrete_sequence=["#03A9F4"])
        st.plotly_chart(fig4, use_container_width=True)
    
    category_feature = st.selectbox("Select a Category Feature", [col for col in selected_features if df[col].dtype == 'object'])
    fig5 = px.bar(df, x=category_feature, y='Churn Prediction', title='Churn Breakdown by Category', color_discrete_sequence=['#9C27B0'])
    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border:1px solid gray">
    <p style="text-align:center;">© 2025 Swift Traq | Designed by Ochieng Festus</p>
    """, unsafe_allow_html=True)
