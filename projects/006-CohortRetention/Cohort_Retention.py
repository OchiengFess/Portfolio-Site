import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import streamlit as st
from PIL import Image
import os 

st.set_page_config(layout="wide", page_title="E-commerce Insights Dashboard", page_icon="📊", initial_sidebar_state="expanded")

base_dir = os.path.dirname(__file__)

data_folder = os.path.join(base_dir, "data")
st.markdown("### Retail Analytics ")
#Load
img = Image.open(os.path.join(data_folder,"retail_shopping.jpg"))
#Resize
img = img.resize((img.width, 900), Image.Resampling.HAMMING)
#Display
st.image(img, use_container_width=True)

df = pd.read_csv(os.path.join(data_folder, "UK_ecommerce_data.csv"), encoding='ISO-8859-1')

# Data Preprocessing
df.dropna(subset=['CustomerID'], inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
#--filter out -ve
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Streamlit UI customization
custom_css = """ 
        <style>
    .main {background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 15px; padding: 20px; box-shadow: 0px 8px 32px rgba(0, 0, 0, 0.2);}
    .stTabs {font-size: 18px; font-weight: bold; color: #ffffff;}
    .stPlotlyChart {border-radius: 12px; box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);}
    .stMarkdown {font-size: 16px; color: #0a2351;}
    .sidebar .sidebar-content {background: #0a192f; color: white; padding: 15px; border-radius: 10px;}
    .loading-spinner {text-align: center; font-size: 18px; font-weight: bold; margin-top: 20px; color: white;}
    footer {text-align: center; padding: 10px; font-size: 14px; color: white; background: rgba(0, 0, 0, 0.3);}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#Sidebar - Filtering Options
#time_period = st.sidebar.selectbox("Select Cohort Period:", ["Monthly", "Weekly", "Quarterly"])
#st.sidebar.success("🛒")
#Load
img = Image.open(os.path.join(data_folder,"bags.jpg"))
#Resize
img = img.resize((img.width, 340), Image.Resampling.HAMMING)
#Display
st.sidebar.image(img, use_container_width=True)
st.sidebar.markdown("#### Filters to Apply")
date_range = st.sidebar.date_input("Select Date Range", [df['InvoiceDate'].min(), df['InvoiceDate'].max()])
category_filter = st.sidebar.multiselect("Filter by Product Category", df['Description'].unique())
region_filter = st.sidebar.multiselect("Filter by Country", df['Country'].unique())

st.sidebar.info("Use the filters above to refine your analysis. Explore sales trends, customer behavior, and geographic distribution with interactive visuals.")

# Apply filters
try:
    # Ensure date_range is valid and has at least 2 elements
    if not date_range or len(date_range) < 2:
        st.warning("⚠️ Please select a valid date range.")
        df_filtered = df.iloc[0:0]  # Return empty DataFrame
    else:
        min_date = df['InvoiceDate'].dt.date.min()
        max_date = df['InvoiceDate'].dt.date.max()

        if date_range[0] > max_date or date_range[1] < min_date:
            st.warning("⚠️ Selected date range is out of available data.")
            df_filtered = df.iloc[0:0]  # Return empty DataFrame
        else:
            df_filtered = df[(df['InvoiceDate'].dt.date >= date_range[0]) & (df['InvoiceDate'].dt.date <= date_range[1])]

except (IndexError, ValueError, TypeError) as e:
    st.error("Error")
    df_filtered = df.iloc[0:0]  # Silently return empty DataFrame without displaying an error
    pass

#df_filtered = df[(df['InvoiceDate'].dt.date >= date_range[0]) & (df['InvoiceDate'].dt.date <= date_range[1])]
if category_filter:
    df_filtered = df_filtered[df_filtered['Description'].isin(category_filter)]
if region_filter:
    df_filtered = df_filtered[df_filtered['Country'].isin(region_filter)]

with st.spinner("Loading Dashboard..."):
    # Tabs
    tabs = st.tabs(["Overview", "Cohort Analysis", "RFM Segmentation", "Geography", "Advanced Insights"])

    # Overview tab
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            df_filtered['Month'] = df_filtered['InvoiceDate'].dt.to_period('M').astype(str)
            monthly_revenue = df_filtered.groupby('Month')['TotalPrice'].sum()
            st.markdown("##### 📈 Monthly Revenue Trends")
            fig = px.line(monthly_revenue, markers=True, labels={'value': 'sales'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

        with col2:
            top_products = df_filtered.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
            st.markdown("##### 🏆 Top 10 Best Selling Products")
            fig = px.bar(top_products, orientation='h', color=top_products.values, color_continuous_scale='blues', labels={'Description': "", 'value': "Qty"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig)

        # Cohort Analysis
        with tabs[1]:
            df_filtered['CohortMonth'] = df_filtered.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
            df_filtered['OrderMonth'] = df_filtered['InvoiceDate'].dt.to_period('M')
            df_filtered['CohortIndex'] = (df_filtered['OrderMonth'] - df_filtered['CohortMonth']).apply(lambda x: x.n)

            cohort_data = df_filtered.groupby(['CohortMonth', 'CohortIndex']).agg(
                n_customers = ('CustomerID', 'nunique')
            ).reset_index()
            cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='n_customers').fillna(0)
            cohort_pivot = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0)

            with st.expander("Retention", expanded=True):
                st.markdown("##### 📊 Customer Retention Cohort Over Time")
                fig, ax = plt.subplots(figsize=(12, 6), dpi=400)
                sns.heatmap(cohort_pivot, annot=True, fmt='.0%', cmap='coolwarm', ax=ax, annot_kws={"size": 8})
                plt.xlabel('CohortIndex [Months]')
                st.pyplot(fig)

            # Customer Lifecycle Heatmap
            revenue_cohort = df_filtered.groupby(['CohortMonth', 'CohortIndex']).agg(
                Revenue=('TotalPrice', 'sum')
            ).reset_index()
            revenue_pivot = revenue_cohort.pivot(index='CohortMonth', columns='CohortIndex', values='Revenue').fillna(0) / 1000.0

            st.markdown("##### 💰 Customer Lifecycle (Revenue per Cohort $K)")
            fig, ax = plt.subplots(figsize=(12, 6), dpi=400)
            sns.heatmap(revenue_pivot, annot=True, fmt='.1f', cmap='coolwarm', ax=ax, annot_kws={'size': 8, 'fontfamily': 'sans-serif'})
            plt.xlabel('CohortIndex [Months]')
            st.pyplot(fig)

        # RFM Segmentation
        with tabs[2]:
            latest_date = df_filtered['InvoiceDate'].max() + pd.DateOffset(1)
            rfm = df_filtered.groupby('CustomerID').agg(
                {
                    'InvoiceDate': lambda x: (latest_date - x.max()).days,
                    'InvoiceNo': 'count',
                    'TotalPrice': 'sum'
                }
            )
            rfm.columns = ['Recency', 'Frequency', 'Monetary']

            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)

            model = KMeans(random_state=42)
            visualizer = KElbowVisualizer(model, k=(2, 10))
            visualizer.fit(rfm_scaled)
            optimal_k = visualizer.elbow_value_ 

            #col1, col2 = st.columns(2)

            with st.expander("Segmentation", expanded=True):
                st.markdown("##### 🎯 Customer Segmentation (RFM Analysis)")
                cluster_options = st.slider("Select No. of Clusters", min_value=2, max_value=10, value=optimal_k)
                kmeans = KMeans(n_clusters=cluster_options, random_state=42)

                rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
                fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', labels={'Recency': 'Recency [Days since Last Purchase]'})
                st.plotly_chart(fig)

            with st.expander("Distribution", expanded=True):
                rfm_counts = rfm['Cluster'].value_counts()
                st.markdown("##### 🔍 Customer Segments Distribution")
                fig = px.pie(names=rfm_counts.index, values=rfm_counts.values, title='Customer Clusters', color_discrete_sequence=px.colors.sequential.Bluyl_r)
                st.plotly_chart(fig)


            # Geography Tab
        with tabs[3]:
            st.markdown("##### 🌍 Sales Distribution by Country")
            country_sales = df_filtered.groupby("Country")["TotalPrice"].sum().reset_index()
            fig = px.choropleth(country_sales, locations="Country", locationmode="country names", color="TotalPrice", labels={'TotalPrice': 'Sales'})
            st.plotly_chart(fig)

            # Advanced Insight 
        with tabs[4]:
            st.markdown("##### 📊 Sales by Hour of the Day")
            #st.write(df_filtered['InvoiceDate'])
            df_filtered['Hour'] = df_filtered['InvoiceDate'].dt.hour
            hourly_sales = df_filtered.groupby('Hour')['TotalPrice'].sum()
            fig = px.bar(hourly_sales, color=hourly_sales.values, color_continuous_scale='blues', labels={'value': 'Sales'})
            st.plotly_chart(fig)

    # Footer
    st.markdown("<hr><center>© 2025 E-commerce Dashboard | DS/DA Ochieng Festus</center>", unsafe_allow_html=True)