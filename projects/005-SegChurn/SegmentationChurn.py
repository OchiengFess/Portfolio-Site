import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from PIL import Image
import requests
from io import BytesIO

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation & Churn Dashboard", layout='centered',initial_sidebar_state='expanded')

#st.image(("https://raw.githubusercontent.com/OchiengFess/Portfolio-Site/main/projects/005-SegChurn/Segmentation_Thumbnail.png"), use_container_width=True)
url = "https://raw.githubusercontent.com/OchiengFess/Portfolio-Site/main/projects/005-SegChurn/Segmentation_Thumbnail2.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Define desired height while keeping full width


img = img.resize((img.width, 700), Image.Resampling.HAMMING)

st.image(img)
# Dark theme settings


base_dir = os.path.dirname(__file__)
data_folder = os.path.join(base_dir, "data")

# Load data
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
def dynamic_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return kmeans

num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
kmeans = dynamic_kmeans(num_clusters)

# Load Best Model for Churn Prediction
try:
    best_model = joblib.load(os.path.join(data_folder, "best_churn_model.pkl"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    best_model = None

#st.title("📊 Customer Segmentation & Churn Analysis")
st.markdown("### 📊 Customer Segmentation & Churn Analysis")
# Sidebar Filters
with st.sidebar:
    st.header("🔍 Filters & Summary")
    date_range = st.date_input("Select Date Range", [df['InvoiceDate'].min(), df['InvoiceDate'].max()])
    df = df[(df['InvoiceDate'] >= pd.to_datetime(date_range[0])) & (df['InvoiceDate'] <= pd.to_datetime(date_range[1]))]
    country_filter = st.multiselect("Filter by Country", df['Country'].unique())
    if country_filter:
        df = df[df['Country'].isin(country_filter)]

# Key Metrics
cols = st.columns(3)
with cols[0]:
    st.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
with cols[1]:
    st.metric("Total Transactions", f"{df.shape[0]:,}")

# Tabs
tabs = st.tabs(["📈 Overview", "📊 Segmentation", "🚀 Churn Prediction", "📉 Insights", "🔍 CLV Analysis"])

with tabs[0]:
    #st.header("Dataset Overview")
    st.markdown("#### Dataset Overview")
    st.dataframe(df.head(6))
    
    st.subheader("Transactions by Country")
    country_counts = df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Transactions']
    fig_country = px.bar(country_counts, x='Country', y='Transactions', title='Transactions per Country')
    st.plotly_chart(fig_country)

with tabs[1]:
    st.markdown("#### Customer Segmentation")
    rfm['Cluster'] = rfm['Cluster'].astype(str)
    cluster_counts = rfm['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_pie = px.pie(cluster_counts, names='Cluster', values='Count', title='Customer Cluster Distribution')
    st.plotly_chart(fig_pie)

    fig_scatter = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Recency vs Monetary')
    st.plotly_chart(fig_scatter)

with tabs[2]:
    st.markdown("#### Predict Customer Churn")
    recency = st.number_input("Recency (Days)", min_value=0, max_value=1000, value=rfm['Recency'].median().astype(int))
    frequency = st.number_input("Frequency", min_value=1, max_value=1000, value=rfm['Frequency'].mean().astype(int))
    monetary = st.number_input("Monetary", min_value=1, max_value=50000, value=rfm['Monetary'].mean().astype(int))
    tenure = st.number_input("Tenure (Days)", min_value=0, max_value=1000, value=90)

    if st.button("Predict Churn") and best_model:
        input_data = np.array([[recency, frequency, monetary, tenure]])
        prediction = best_model.predict(input_data)[0]
        churn_probability = best_model.predict_proba(input_data)[0,1]
        st.metric("Churn Probability", f"{churn_probability:.2%}")
        st.write("Prediction: ", "🔴 Likely to Churn" if prediction else "🟢 Retained")

with tabs[3]:
    st.markdown(" #### 📉 Additional Insights")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period("M").astype(str)
    retention = df.groupby('YearMonth')['CustomerID'].nunique().reset_index()
    fig_retention = px.line(retention, x='YearMonth', y='CustomerID', 
                            title='Customer Retention Over Time',
                            labels={'CustomerID': 'No. of Customers', 'YearMonth': 'Month-Year'})
    st.plotly_chart(fig_retention)

with tabs[4]:
    st.markdown("#### 🔍  Customer Lifetime Value (CLV) Analysis")
    clv = df.groupby('CustomerID').agg({'TotalPrice': 'sum'}).reset_index()
    clv.columns = ['CustomerID', 'CLV']
    fig_clv = px.histogram(clv, x='CLV', nbins=30, title='Distribution of Customer Lifetime Value')
    st.plotly_chart(fig_clv)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Swift Traq**")
st.sidebar.markdown("**📊Turning Raw Data into Business Intelligence**")
st.sidebar.markdown(
    "Swift Traq is a cutting-edge data analytics consultancy, specializing in **predictive analytics, customer insights, and revenue optimization.**"
)
st.sidebar.markdown(
    "We help businesses **decode data, uncover trends, and drive smarter decisions** with powerful analytics solutions."
)
st.sidebar.markdown("---")  # Separator for clarity


# Footer
st.markdown("""
    <hr style="border:1px solid gray">
    <p style="text-align:center;">© 2025 Swift Traq | Designed by Ochieng Festus</p>
    """, unsafe_allow_html=True)
