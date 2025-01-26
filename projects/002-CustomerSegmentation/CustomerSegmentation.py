import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

#----------Streamlit Config -----------
st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="🔍", initial_sidebar_state='expanded')
st.title("Customer Segmentation Dashboard")

#Get base directory
base_dir = os.path.dirname(__file__)

# Build path to data folder
data_folder = os.path.join(base_dir, "data")

# Load Model and Scaler
kmeans_model = joblib.load(os.path.join(data_folder, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(data_folder, "scaler.pkl"))

@st.cache_data
def data(file_name: str, chunk_size=1000):
     return pd.concat(pd.read_csv(os.path.join(data_folder, file_name), chunksize=chunk_size, delimiter="\t"), ignore_index=True)

# Load wrangled data
def load_data() -> pd.DataFrame:
    df_customer = data('Clean_DimCustomer.csv')
    df_sales = data('Clean_FactInternetSales.csv')
    df_date = data('Clean_DimDate.csv')
    df_geography = data('DimGeography.csv')

    df_sales['OrderDate'] = pd.to_datetime(df_sales['OrderDate'])
    df_date['FullDateAlternateKey'] = pd.to_datetime(df_date['FullDateAlternateKey'])

    df_customer['FullName'] = df_customer['FirstName'] + ' ' + df_customer['LastName']
    df = pd.merge(df_sales, df_customer, on='CustomerKey')
    df = pd.merge(df, df_date, left_on='OrderDateKey', right_on='DateKey')

    df_clv = df.groupby('CustomerKey').agg(
        TotalRevenue=('SalesAmount', 'sum'),
        TotalOrders=('SalesOrderNumber', 'nunique'),
        AvgOrderValue=('SalesAmount', 'mean'),
        LastPurchaseDate=('OrderDate', 'max'),
        FirstPurchaseDate=('OrderDate', 'min')
    ).reset_index()

    df_clv['CustomerLifetime'] = (df_clv['LastPurchaseDate'] - df_clv['FirstPurchaseDate']).dt.days
    df_clv = pd.merge(df_clv, df_customer[['CustomerKey', 'GeographyKey', 'FullName']], on='CustomerKey')
    df_clv = pd.merge(df_clv, df_geography[['GeographyKey', 'EnglishCountryRegionName']], on='GeographyKey')

    return df_clv

# Prepare data
data = load_data()
data_features = data[['TotalRevenue', 'TotalOrders', 'AvgOrderValue', 'CustomerLifetime']].dropna()
data_scaled = scaler.transform(data_features)
data['Cluster'] = kmeans_model.predict(data_scaled)

# PCS for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

#----------Streamlit App -------------------------------

menu = st.sidebar.selectbox("Menu", ["Overview", "Visualizations", "Cluster Insights"])

if menu == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(data))
    col2.metric("Total Revenue", f"${data['TotalRevenue'].sum():,.2f}")
    col3.metric("Total Order Value", f"${data['AvgOrderValue'].mean():,.2f}")

    st.subheader("Dataset Overview")
    st.dataframe(data.head(6))


elif menu == 'Visualizations':
     st.subheader('Cluster visualizations')
     tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "PCA Clusters", 
        "Revenue Contribution", 
        "Cluster Composition", 
        "Geographical Insights", 
        "Customer Lifetime Distribution"
    ])
     
     with tab1:
          st.subheader("PCA Cluster visualization")
          fig_pca = px.scatter(
               data,
               x='PCA1',
               y='PCA2',
               color='Cluster',
               title="Clusters (PCA Reduced Dimensions) ",
               labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
               color_continuous_scale=px.colors.qualitative.Set1       
          )
          st.plotly_chart(fig_pca)
    
     with tab2:
        st.subheader("Revenue Contribution by Cluster")
        revenue_by_cluster = data.groupby('Cluster')['TotalRevenue'].sum().reset_index()
        fig_revenue = px.bar(
            revenue_by_cluster,
            x='Cluster',
            y='TotalRevenue',
            title='Total Revenue by Cluster',
            labels={'TotalRevenue': 'Revenue', 'Cluster': 'Cluster'}
        )
        st.plotly_chart(fig_revenue)

     with tab3:
         st.subheader('Cluster Composition')
         fig_box = px.box(
             data,
             x='Cluster',
             y='TotalRevenue',
             color='Cluster',
             title='Revenue Distribution  by Cluster',
             labels={'TotalRevenue': 'Revenue', 'Cluster': 'Cluster'},
             color_discrete_sequence=px.colors.qualitative.Set1
         )
         st.plotly_chart(fig_box)
    
     with tab4:
       geo_revenue = data.groupby('EnglishCountryRegionName')['TotalRevenue'].sum().reset_index()
       fig_geo = px.choropleth(
           geo_revenue,
           locations='EnglishCountryRegionName',
           locationmode='country names',
           color='TotalRevenue',
           title='Revenue Distribution by Region',
           color_continuous_scale= px.colors.sequential.Plasma

       )  
       st.plotly_chart(fig_geo)

     with tab5:
         fig_hist = px.histogram(
             data,
             x='CustomerLifetime',
             color='Cluster',
             title='Customer Lifetime Distribution by Cluster',
             nbins=20
         )
         st.plotly_chart(fig_hist)
    

elif menu == 'Cluster Insights':
    st.subheader('Detailed cluster Insights')
    cluster_metrics = data.groupby('Cluster').agg(
        {
            'TotalRevenue': 'mean',
            'TotalOrders': 'mean',
            'AvgOrderValue': 'mean',
            'CustomerLifetime': 'mean'
        }
    ).round(2)
    st.dataframe(cluster_metrics)
    st.bar_chart(cluster_metrics, color=['#002D62', '#007791', '#1E90FF', '#ADD8E6'])


st.sidebar.success("Use menu to navigate through the dashboard.")

# Footer
st.markdown("---")
st.markdown("Data Scientist/Data Analyst: Ochieng Festus")
st.markdown("*Powered by ©️ Swift Traq* | 2025")
    

     