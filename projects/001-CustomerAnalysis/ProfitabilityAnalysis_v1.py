import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configurations
st.set_page_config(page_title="Profitability Analysis Dashboard", layout="wide", page_icon="📊")

# Custom Styles
custom_palette = {
    "primary": "#1E3A8A",
    "secondary": "#3B82F6",
    "accent": "#93C5FD"
}

def apply_custom_styles():
    st.markdown(
        f"""
        <style>
        .css-1d391kg {{
            background-color: {custom_palette['primary']} !important;
        }}
        .stMetric-value {{
            color: {custom_palette['accent']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    apply_custom_styles()

# Load data
@st.cache_data
def load_data(file_path, chunk_size=1000):
    df_customer = pd.concat(pd.read_csv(file_path+'\Clean_DimCustomer.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_sales = pd.concat(pd.read_csv(file_path+'\Clean_FactInternetSales.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_dimProduct = pd.concat(pd.read_csv(file_path+'\Clean_DimProduct.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_dimProductSubcategory = pd.concat(pd.read_csv(file_path+'\Clean_DimProductsSubcategory.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_date = pd.concat(pd.read_csv(file_path+'\Clean_DimDate.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_geography = pd.concat(pd.read_csv(file_path+'\DimGeography.csv', chunksize=chunk_size, delimiter="\t"), ignore_index=True)

    df_sales['OrderDate'] = pd.to_datetime(df_sales['OrderDate'])
    df_date['FullDateAlternateKey'] = pd.to_datetime(df_date['FullDateAlternateKey'])

    df = pd.merge(df_sales, df_customer, on='CustomerKey')
    df = pd.merge(df, df_dimProduct[['ProductKey','ProductSubcategoryKey','EnglishProductName']], on='ProductKey')
    df = pd.merge(df, df_dimProductSubcategory[['ProductSubcategoryKey','EnglishProductSubcategoryName']], on='ProductSubcategoryKey')
    df = pd.merge(df, df_geography[['GeographyKey','City','StateProvinceName','CountryRegionCode','EnglishCountryRegionName']], on='GeographyKey')
    df = pd.merge(df, df_date, left_on='OrderDateKey', right_on='DateKey')

    df['Profit'] = df['SalesAmount'] - df['TotalProductCost'] - df['TaxAmt']
    df['ProfitMargin'] = (df['Profit'] / df['SalesAmount'])* 100

    return df

file_path = r"D:\STREAMLIT\008_CustomerAnalysis\data"
data = load_data(file_path)

# Metrics
st.title("Profitability Analysis Dashboard")
st.markdown("**Insights into Key profitability metrics and trends**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = data['SalesAmount'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")

with col2:
    total_profit = data['Profit'].sum()
    st.metric("Total Profit", f"${total_profit:,.2f}")

with col3:
    avg_profit_margin = data['ProfitMargin'].mean()
    st.metric("Avg. Profit Margin", f"{avg_profit_margin:.2f}%")

with col4:
    total_orders = data['SalesOrderNumber'].nunique()
    st.metric("Total Orders", f"{total_orders:,}")

# Tabs for analysis
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview","Revenue Trends", "Profit Margins", "Regional Insights"])

with tab1:
    with st.expander("Multi-Table Data  --", expanded=True):
        st.dataframe(data.head())


with tab2:
    st.subheader("Revenue and Profit Trends Over Time")

    monthly_data = data.groupby(data['OrderDate'].dt.to_period('M')).agg(
        Revenue=('SalesAmount', 'sum'),
        Profit=('Profit', 'sum')
    ).reset_index()
    monthly_data['OrderDate'] = monthly_data['OrderDate'].dt.to_timestamp()

    fig = px.line(monthly_data,
                  y=['Revenue', 'Profit'],
                  title="Monthly Revenue and Profit Trends",
                  labels={'value': 'Amount', 'variable': 'Metric'},
                  template='plotly_dark')
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Profit Margin Distribution")

    fig2 = px.histogram(data,
                        x='ProfitMargin',
                        nbins=30,
                        title="Profit Margin Distribution",
                        labels={'ProfitMargin': 'Profit Margin (%)'},
                        template='plotly_dark',
                        color_discrete_sequence=[custom_palette['secondary']])
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    ) 
    st.plotly_chart(fig2)     

    st.subheader("Top 10 Most Profitable Products")
    top_products = data.groupby(['ProductKey','EnglishProductName']).agg(
        TotalProfit=('Profit', 'sum')
    ).nlargest(10, 'TotalProfit').reset_index()

    fig3 = px.bar(top_products,
                  x='EnglishProductName',
                  y='TotalProfit',
                  labels={'EnglishProductName': 'Product', 'TotalProfit': 'Total Profit'},
                  template="plotly_dark",
                  color='TotalProfit',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Regional Performance Insights")

    regional_data = data.groupby('EnglishCountryRegionName').agg(
        TotalRevenue=('SalesAmount', 'sum'),
        TotalProfit=('Profit', 'sum'),
        AvgProfitMargin=('ProfitMargin', 'mean')
    ).reset_index()

    fig4 = px.bar(regional_data,
                  x='EnglishCountryRegionName',
                  y='TotalRevenue',
                  title='Revenue by Region',
                  labels={'EnglishCountryRegionName': 'Region','TotalRevenue': 'Total Revenue'},
                  color='TotalRevenue',
                  template='plotly_dark',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(regional_data,
                      x='TotalRevenue',
                      y='TotalProfit',
                      size='AvgProfitMargin',
                      color='EnglishCountryRegionName',
                      title='Revenue vs Profit by Region',
                      labels={'TotalRevenue': 'Total Revenue', 'TotalProfit': 'Total Profit'},
                      template='plotly_dark')
    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Data Scientist/Data Analyst: Ochieng Festus")
st.markdown("*Powered by ©️ Swift Traq* | 2025")