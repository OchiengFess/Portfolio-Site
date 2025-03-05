import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation & Churn Dashboard", layout='centered')

base_dir = os.path.dirname(__file__)
data_folder = os.path.join(base_dir, "data")

#Load data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(data_folder, "UK_ecommerce_data.csv"), 
                     encoding='ISO-8859-1', 
                     low_memory=False,
                     dtype={'CustomerID': str})
    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# Compute RFM Metrics
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg(
    {
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }
)
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Feature Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply K-Means Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# LOad Best Model for Churn Prediction
try:
    best_model = joblib.load(os.path.join(data_folder, "best_churn_model.pkl"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    best_model = None

# Streamlit UI
st.title("📊 Customer Segmentation & Churn Analysis")

# Sidebar with Filters
with st.sidebar:
    st.header("🔍 Filters & Summary")
    country_filter = st.multiselect("Filter by Country", df['Country'].unique())
    if country_filter:
        df = df[df['Country'].isin(country_filter)]

cols = st.columns(3)

with cols[0]:
    st.metric("Total Customers", f"{df['CustomerID'].nunique():,}")

with cols[1]:
    st.metric("Total Transactions", f"{df.shape[0]:,}")

# Tabs
tabs = st.tabs(["📈 Overview", "📊 Segmentation", "🚀 Churn Prediction", "📉 Insights"])

with tabs[0]:
    st.header("Dataset Overview")
    st.dataframe(df.head(6))

    st.subheader("Transactions by Country")
    country_counts = df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Transactions']
    fig_country = px.bar(country_counts, x='Country', 
                         y='Transactions',
                         title='Transactions per country',
                         height=500
                         )
    st.plotly_chart(fig_country)

    with tabs[1]:
        st.header("Customer Segmentation")

        rfm['Cluster'] = rfm['Cluster'].astype(str)
        cluster_counts = rfm['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(cluster_counts, names='Cluster',
                         values='Count',
                         title='Customer Cluster Distribution',
                         color_discrete_sequence=px.colors.qualitative.Set1)
        
        st.plotly_chart(fig_pie)

        fig_scatter = px.scatter(rfm, x='Recency',
                                 y='Monetary',
                                 color='Cluster',
                                 title='Customer Segmentation: Recency vs Monetary',
                                 hover_data=['Frequency'])
        st.plotly_chart(fig_scatter)

        with tabs[2]:
            st.header("Predict Customer Churn")

            recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=1000, value=rfm['Recency'].median().astype(int))
            frequency = st.number_input("Frequency (Number of purchases)", min_value=1, max_value=1000, value=rfm['Frequency'].mean().astype(int))
            monetary = st.number_input("Monetary (Total Spend)", min_value=1, max_value=50000, value=rfm['Monetary'].mean().astype(int))
            tenure = st.number_input("Tenure (Days since First Purchase)", min_value=0, max_value=1000, value=90)

            if st.button("Predict Churn") and best_model:
                input_data = np.array([[recency, frequency, monetary, tenure]])
                #input_data = scaler.transform(input_data)
                prediction = best_model.predict(input_data)[0]
                churn_probability = best_model.predict_proba(input_data)[0,1]

                fig_gauge = px.bar_polar(
                    r=[churn_probability * 100],
                    theta=["Churn Probability"],
                    range_r =[0, 100],
                    title="Churn Probability",
                    color_discrete_sequence=["red"] if churn_probability > 0.5 else ["green"]

                )
                st.metric(label="Churn Probability", value=f"{churn_probability:.2%}")
                st.write("Churn Prediction: ", "🔴 Likely to Churn" if prediction else "🟢 Retained")
                st.plotly_chart(fig_gauge)

        with tabs[3]:
            st.header("📉 Additional Insights")

            #feature_importance = pd.Series(best_model.feature_importances_,
                                           #index=['Recency','Frequency','Monetary', 'Tenure'])
            #fig_feature_importance = px.bar(feature_importance,
                                            #x=feature_importance.index,
                                            #y=feature_importance.values,
                                            #title='Feature Importance in Churn Prediction'
                                            #)
            #st.plotly_chart(fig_feature_importance)

            df['YearMonth'] = df['InvoiceDate'].dt.to_period("M").astype(str)
            retention = df.groupby('YearMonth')['CustomerID'].nunique().reset_index()
            fig_retention = px.line(retention,
                                    x='YearMonth',
                                    y='CustomerID',
                                    title='Customer Retention Over Time',
                                    labels={'CustomerID': 'No. of Customers', 'YearMonth': 'Year-Month'})
            st.plotly_chart(fig_retention)