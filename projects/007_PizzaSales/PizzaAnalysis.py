import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
import os

st.set_page_config("Pizza Sales Analysis", page_icon="🍕")

base_dir = os.path.dirname(__file__)

data_folder = os.path.join(base_dir, "data")

#-- custom CSS
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

# Load CSV files
orders = pd.read_csv(os.path.join(data_folder, "orders.csv"))
order_details = pd.read_csv(os.path.join(data_folder, "order_details.csv"))
pizzas = pd.read_csv(os.path.join(data_folder, "pizzas.csv"))
pizza_types = pd.read_csv(os.path.join(data_folder, "pizza_types.csv"), encoding='ISO-8859-1')

#convert date & time columns
orders['date'] = pd.to_datetime(orders['date'])
orders['time'] = pd.to_datetime(orders['time'], format='%H:%M:%S').dt.time

# Merge datasets
order_details = order_details.merge(pizzas, on='pizza_id', how='left')
order_details = order_details.merge(pizza_types, on='pizza_type_id', how='left')
full_data = order_details.merge(orders, on='order_id', how='left')

# Compute total revenue
full_data['total_price'] = full_data['quantity'] * full_data['price']
total_revenue = full_data['total_price'].sum()

# Compute total number of  orders
total_orders = full_data['order_id'].nunique()

# Streamlit UI
st.title("🍕Pizza Sales Analytics")

# Metrics Overview
col1, col2 = st.columns(2)
col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales Overview", "Revenue Trends", "Customer Insights", "Market Segmentation", "Market Basket Analysis"])

# category-Wise Sales Analysis
category_sales = full_data.groupby('category')['total_price'].sum().reset_index().sort_values(by='total_price', ascending=False)
fig_category = px.bar(category_sales, x='total_price', y='category', 
                      title='Sales by Category', color='total_price', 
                      color_continuous_scale=['#ffcccc', '#960018'],
                      labels={'total_price': 'total revenue', 'category': ''})

with tab1:
    st.markdown('##### Sales by Category')
    st.plotly_chart(fig_category, use_container_width=True)

# Top selling Pizzas
top_pizzas = full_data.groupby('name')['total_price'].sum().reset_index().sort_values(by='total_price', ascending=False).head(10)
fig_top_pizzas = px.bar(top_pizzas, x='total_price', y='name', 
                        title='Top 10 Best Selling Pizzas', 
                        color='total_price', color_continuous_scale= ['#ffcccc', '#960018'],
                        labels={'total_price': 'revenue', 'name': ''})

with tab1:
    st.markdown('##### Top-Selling Pizzas')
    st.plotly_chart(fig_top_pizzas, use_container_width=True)


# Sales by weekday
full_data['weekday'] = pd.to_datetime(full_data['date']).dt.day_name()
weekday_sales = full_data.groupby('weekday')['total_price'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
fig_weekly_sales = px.bar(weekday_sales, x='weekday',
                          y='total_price',
                          title='Sales by Weekday',
                          color='total_price',
                          color_continuous_scale=['#ffcccc', '#960018'],
                          labels={'weekday': '', 'total_price': 'revenue'}
                          )

with tab2:
    st.markdown('##### Sales by Weekday')
    st.plotly_chart(fig_weekly_sales, use_container_width=True)

# Hourly Sales Trends
full_data['hour'] = pd.to_datetime(full_data['time'], format='%H:%M:%S').dt.hour
hourly_sales = full_data.groupby('hour')['total_price'].sum().reset_index()
fig_hourly_sales = px.bar(hourly_sales, x='hour',
                          y='total_price',
                          title='Hourly Sales Trend',
                          color='total_price',
                          color_continuous_scale=['#ffcccc', '#960018']
                          )
with tab2:
    st.markdown('##### Hourly Sales Trend')
    st.plotly_chart(fig_hourly_sales, use_container_width=True)

# Order size distribution
order_size = full_data.groupby('order_id')['quantity'].sum()
fig_order_size = px.histogram(order_size, nbins=20, title='Order size Distribution', color_discrete_sequence=['#960018'])

with tab3:
    st.markdown('##### Order Size Distribution')
    st.plotly_chart(fig_order_size, use_container_width=True)

# Price sensitivity
price_sensitivity = full_data.groupby('order_id').agg(
    {
        'total_price': 'sum',
        'price': 'mean'
    }
).reset_index()
fig_price_sensitivity = px.scatter(price_sensitivity, x='price',
                                   y='total_price',
                                   title='Price Sensitivity Analysis',
                                   labels={'price': 'Average Price Per Item', 'total_price': 'Total Spending'},
                                   color_discrete_sequence=['#960018'])

with tab3:
    st.markdown('##### Price Sensitivity Analysis')
    st.plotly_chart(fig_price_sensitivity, use_container_width=True)

# Revenue Trends Over Time
full_data['month'] = full_data['date'].dt.to_period('M').dt.to_timestamp()
revenue_trend = full_data.groupby('month')['total_price'].sum().reset_index()
fig_revenue = px.line(revenue_trend, x='month', y='total_price',
                      title='Monthly Revenue Trend',
                      markers=True,
                      labels={'total_price': 'revenue'})
fig_revenue.update_traces(line_color='#960018', marker=dict(color='#960018'))

with tab2:
    st.markdown('##### Monthly Revenue Trends')
    st.plotly_chart(fig_revenue, use_container_width=True)

# Customer buying patterns
size_distribution = full_data.groupby('size')['quantity'].sum().reset_index()
fig_size = px.pie(size_distribution, values='quantity', names='size',
                  title='Pizza Size Preference',
                  color_discrete_sequence=['#960018'])

with tab3:
    st.markdown('##### Pizza Size Preferences')
    st.plotly_chart(fig_size, use_container_width=True)

# Customer segmentation
customer_data = full_data.groupby('order_id')['total_price'].sum().reset_index()
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data)
fig_segment = px.scatter(customer_data, x=customer_data.index, 
                         y=customer_data['total_price'],
                         color=customer_data['cluster'],
                         title='Customer Segmentation based on Spend($)',
                         color_continuous_scale='magma',
                         labels={'category': 'Spend [$]'})

with tab4:
    st.markdown('##### Customer Segmentation')
    st.plotly_chart(fig_segment)

    # Footer
st.markdown("<hr>"
                "<center><em>DS/DA Ochieng Fess</em></center>"
                
                "<center>© 2025 E-commerce Dashboard "
                "</center>", unsafe_allow_html=True)