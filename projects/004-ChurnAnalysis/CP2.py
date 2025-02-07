import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import os

st.set_page_config("Churn Prediction", page_icon="📉")
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

# Load training data to initialize label encoders
df_train = pd.read_csv(os.path.join(data_folder, "preprocessed_v1.csv"))
label_encoders = {col: LabelEncoder().fit(df_train[col]) for col in df_train.select_dtypes(include=['object']).columns}

# Sidebar
st.sidebar.header("Churn Prediction Dashboard")
st.sidebar.subheader("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Tabs for navigation
tabs = st.tabs(["Train Data Preview", "Manual Prediction", "Batch Prediction", "Insights & Visualizations"])

# Train Data Preview
with tabs[0]:
    st.subheader("Train Data Sample")
    import streamlit as st

    st.info(
        "🔍 **Customer Churn Classification:**\n\n"
        "- Customers whose last order was more than **6 months (180 days) ago** are classified as **Churned** 🛑.\n"
        "- Active customers made a purchase **within the last 6 months** ✅.\n\n"
        "This helps identify customers at risk and design strategies to re-engage them. 📊📢"
    )

    
    original_data = pd.read_csv(os.path.join(data_folder, "preprocessed_v1.csv"))
    st.dataframe(original_data.head(6))

    fig = px.histogram(original_data, x='Recency', title="Recency Distribution: Days Since Last Purchase", nbins=30)
    st.plotly_chart(fig)

    fig = px.histogram(original_data, x='Frequency', title="Frequency Distribution: No. of Transactions Per Customer", nbins=30)
    st.plotly_chart(fig)

    #fig = px.box(original_data, x='Churn', y='MonetaryValue', title="Monetary Value by Churn Status")
    #fig.show()

with tabs[1]:
    st.subheader("Manual Churn Prediction")
    user_input = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(selected_features):
        default_value = 0
        if df_train[feature].dtype == 'object':
            options = df_train[feature].dropna().unique().tolist()
            default_value = options[0] if options else "Unknown"
            user_input[feature] = col1.selectbox(f"{feature}", options, index=0) if i % 2 == 0 else col2.selectbox(f"{feature}", options, index=0)
        elif np.issubdtype(df_train[feature].dtype, np.datetime64):
            default_value = df_train[feature].min()
            user_input[feature] = col1.date_input(f"{feature}", value=default_value) if i % 2 == 0 else col2.date_input(f"{feature}", value=default_value)
        else:
            default_value = df_train[feature].median()
            user_input[feature] = col1.number_input(f"{feature}", value=default_value) if i % 2 == 0 else col2.number_input(f"{feature}", value=default_value)

    if st.button("Predict Churn"):
        user_df = pd.DataFrame([user_input])
        for col, encoder in label_encoders.items():
            if col in user_df.columns:
                user_df[col] = encoder.transform(user_df[col])
        user_scaled = scaler.transform(user_df)
        churn_prob = model.predict_proba(user_scaled)[:, 1][0] 
        churn_pred = model.predict(user_scaled)[0]

        st.success(f"**Churn Probability: {churn_prob:.2f}")
        st.success(f"**Churn Prediction: {'Yes, customer is predicted to Churn' if churn_pred == 1 else 'No, customer retained'}")

with tabs[2]:
    st.subheader("Batch Churn Prediction")
    st.info("Upload your csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("##### Uploaded Data Sample")
        st.dataframe(df.head())
    else:
        df_batch = df_train.copy()
        st.write("##### Prediction Data Sample ")
        st.dataframe(df_batch[selected_features].head())
    
    # Preprocess data
    df_selected = df_batch[selected_features].copy()
    for col, encoder in label_encoders.items():
        if col in df_selected.columns:
            df_selected[col] = encoder.transform(df_selected[col])
    df_scaled = scaler.transform(df_selected)

    # Predict Churn
    churn_probs = model.predict_proba(df_scaled)[:, 1]
    df_batch['Churn Probability'] = churn_probs
    df_batch['Churn Prediction'] = np.where(churn_probs > 0.5, 1, 0)
    df_batch['Churn Prediction[Normalized]'] = np.where(df_batch['Churn Prediction'] == 1, "Yes", "No")

    # Display Predictions
    st.write("##### Churn Predictions")
    st.dataframe(df_batch[['Churn Probability', 'Churn Prediction', 'Churn Prediction[Normalized]']].head())

    # Churn distribution
    #st.write("##### Churn Prediction Distribution")
    fig = px.histogram(df_batch, x='Churn Probability', title='Churn Distribution', nbins=2, color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig)


# Insights & Visualizations
with tabs[3]:
    st.subheader("Insights & Visualizations")
    df = df_batch if uploaded_file is None else df

    visualize = st.checkbox("Generate visuals after Prediction")
    if visualize:
        # Churn Rate over time
        if 'OrderDate' in df.columns:
            df['OrderDate'] = pd.to_datetime(df['OrderDate'])
            churn_rate = df.groupby(df['OrderDate'].dt.to_period("M"))['Churn Prediction'].mean().reset_index()
            churn_rate['OrderDate'] = churn_rate['OrderDate'].astype(str)
            fig = px.line(churn_rate, x='OrderDate', y='Churn Prediction', title="Churn Rate Over Time", color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig)

        # Churn breakdown by Category
        category_feature = st.selectbox("Select a Category feature", [col for col in selected_features if df[col].dtype == 'object'])
        fig = px.bar(df, x=category_feature, y='Churn Prediction', title='Churn Breakdown by Category', color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig)
    else:
        st.info("Enable the checkbox to generate Insights")

# Add footer
st.markdown("""
    <hr style="border:1px solid gray">
    <p style="text-align:center;">DS/DA:OChieng Festus  | © Copyright by Swift Traq</p>
""", unsafe_allow_html=True)
