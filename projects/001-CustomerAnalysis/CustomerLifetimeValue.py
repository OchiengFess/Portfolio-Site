import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Lifetime Value Dashboard", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Get base directory
base_dir = os.path.dirname(__file__)

# Build path to data folder
data_folder = os.path.join(base_dir, "data")

@st.cache_data
def load_data(file_name, chunk_size=1000) -> pd.DataFrame:
    df_customer = pd.concat(pd.read_csv(os.path.join(data_folder, "Clean_DimCustomer.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_sales = pd.concat(pd.read_csv(os.path.join(data_folder, "Clean_FactInternetSales.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_date = pd.concat(pd.read_csv(os.path.join(data_folder,"Clean_DimDate.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_geography = pd.concat(pd.read_csv(os.path.join(data_folder, "DimGeography.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)

    df_sales['OrderDate'] = pd.to_datetime(df_sales['OrderDate'])
    df_date['FullDateAlternateKey'] = pd.to_datetime(df_date['FullDateAlternateKey'])

    df_customer['FullName'] = df_customer['FirstName']+ ' '+ df_customer['LastName']
    df = pd.merge(df_sales, df_customer, on='CustomerKey')
    df = pd.merge(df, df_date, left_on='OrderDateKey', right_on='DateKey')

    # CLV related metrics
    df_clv = df.groupby('CustomerKey').agg(
        TotalRevenue=('SalesAmount', 'sum'),
        TotalOrders=('SalesOrderNumber', 'nunique'),
        AvgOrderValue=('SalesAmount', 'mean'),
        LastPurchaseDate=('OrderDate', 'max'),
        FirstPurchaseDate=('OrderDate', 'min')
    ).reset_index()

    df_clv['CustomerLifetime'] = (df_clv['LastPurchaseDate'] - df_clv['FirstPurchaseDate']).dt.days
    df_clv = pd.merge(df_clv, df_customer[['CustomerKey','GeographyKey', 'FullName']], on='CustomerKey')
    df_clv = pd.merge(df_clv, df_geography[['GeographyKey','EnglishCountryRegionName']], on='GeographyKey')

    return df_clv

file_path = r"D:\STREAMLIT\008_CustomerAnalysis\data"
data = load_data(file_path)

# Define color palette
CUSTOM_BLUE = '#007BFF'
BACKGROUND_COLOR = '#F9F9F9'

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.selectbox("Select Region", ["All"] + list(data['EnglishCountryRegionName'].unique()))
filtered_data = data if selected_region == 'All' else data[data['EnglishCountryRegionName'] == selected_region]

# Dashboard title
st.title("Customer Lifetime Value dashboard")
st.markdown("**Presented by Swift Traq**")

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Revenue",
              f"${filtered_data['TotalRevenue'].sum():,.2f}")
    
with col2:
    st.metric("Average Order value",
              f"${filtered_data['AvgOrderValue'].mean():,.2f}")
    
with col3:
    st.metric("Average Customer Lifetime",
              f"{filtered_data['CustomerLifetime'].mean():.0f}")
    
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Lifetime Distribution", "Revenue by Region", "Order Value Distribution", "Top Customers"])

# Lifetime distribution
with tab1:
    st.subheader("Customer Lifetime Distribution")
    fig1 = px.histogram(
        filtered_data,
        x='CustomerLifetime',
        nbins=30,
        title='Distribution of Customer Lifetime (in days)',
        template='plotly_white',
        color_discrete_sequence=["#002D62"]
    )
    fig1.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        bargap=0.2
    )
    st.plotly_chart(fig1, use_container_width=True)

# Revenue by Region
with tab2:
    st.subheader("Revenue contribution by Region")
    region_revenue = filtered_data.groupby('EnglishCountryRegionName')['TotalRevenue'].sum().reset_index()
    fig2 = px.bar(
        region_revenue,
        x='EnglishCountryRegionName',
        y='TotalRevenue',
        title='Revenue Contribution by Region',
        template='plotly_white',
        color='TotalRevenue',
        color_continuous_scale='blues'
    )
    fig2.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig2, use_container_width=True)

# Order value Distribution
with tab3:
    st.subheader("Average Order value Distribution")
    fig3 = px.box(
        filtered_data,
        y='AvgOrderValue',
        title='Distribution of Average Order value',
        template='plotly_white',
        color_discrete_sequence=["#002D62"]
    )
    fig3.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig3, use_container_width=True)

# Top Customers
with tab4:
    st.subheader("Top 10 Customers by Revenue")
    top_customers = filtered_data.nlargest(10, 'TotalRevenue').sort_values(by='TotalRevenue', ascending=False)
    fig4 = px.bar(
        top_customers,
        y='TotalRevenue',
        x='FullName',
        title='Top 10 Customers by Revenue',
        color='TotalRevenue',
        color_continuous_scale='Blues'
    )
    fig4.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Data Scientist/Data Analyst: Ochieng Festus")
st.markdown("*Powered by ©️ Swift Traq* | 2025")