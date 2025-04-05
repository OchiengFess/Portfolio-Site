import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import os

st.set_page_config("Pizza Sales Analysis", page_icon="🍕", layout='wide')

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
    #footer {text-align: center; padding: 10px; font-size: 14px; color: white; background: rgba(0, 0, 0, 0.3);}
    
        .stPlotlyChart:hover {
        transform: scale(1.01);
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .stMetric {
        background: #DB7093;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        color: white;
    }
        div[data-testid="stMetric"] > div {
        color: white !important;
    }

    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

#Load
img = Image.open(os.path.join(data_folder,"pizzas_02.jpg"))
#Resize
img = img.resize((img.width, 300), Image.Resampling.HAMMING)
#Display
#st.image(img, use_container_width=True)

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


# Streamlit UI
st.title("🍕Pizza Sales Analytics")


# filters
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('##### 📅 Filter by Date Range')
    min_date = full_data['date'].min()
    max_date = full_data['date'].max()
    date_range = st.date_input("Select date range",[min_date, max_date], min_value=min_date, max_value=max_date)

# filter by date
if len(date_range) !=2:
    st.warning("Please select valid start & end date")
    filtered_data = full_data
else:
    start_date, end_date = date_range
    filtered_data = full_data[(full_data['date'] >= pd.to_datetime(start_date)) & (full_data['date'] <= pd.to_datetime(end_date))]
    filtered_dataCopy = filtered_data.copy()

# Category Level Drill down
with col2:
    category_options = ['All'] + list(full_data['category'].unique())
    selected_category = st.selectbox("🍕 Filter Top Pizzas by Category", category_options)

if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['category'] == selected_category]
else:
    filtered_data = filtered_dataCopy

with col3:
    weekday_options = ['All', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_weekday = st.selectbox("Choose Weekday",weekday_options)
if selected_weekday != 'All':
    filtered_data = filtered_data[pd.to_datetime(filtered_data['date']).dt.day_name() == selected_weekday]
else:
    filtered_data = filtered_dataCopy

# total revenue --filtered data
total_revenue = filtered_data['total_price'].sum()

# Compute total number of  orders -- filtered data
total_orders = filtered_data['order_id'].nunique()

# Average order value
avg_order_value = total_revenue / total_orders if total_orders != 0 else 0

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Order Value", f"{avg_order_value:.2f}")

# category-Wise Sales Analysis
category_sales = filtered_data.groupby('category')['total_price'].sum().reset_index().sort_values(by='total_price', ascending=False)
fig_category = px.bar(category_sales, x='total_price', y='category', 
                      title='Sales by Category', color='total_price', 
                      color_continuous_scale=['#ffcccc', '#960018'],
                      labels={'total_price': 'total revenue', 'category': ''})
fig_category.update_coloraxes(showscale=False)

# Top selling Pizzas
top_pizzas = filtered_data.groupby('name')['total_price'].sum().reset_index().sort_values(by='total_price', ascending=False).head(10)
fig_top_pizzas = px.bar(top_pizzas, x='total_price', y='name', 
                        title='Top 10 Best Selling Pizzas', 
                        color='total_price', color_continuous_scale= ['#ffcccc', '#960018'],
                        labels={'total_price': 'revenue', 'name': ''})
fig_top_pizzas.update_coloraxes(showscale=False)


# Sales by weekday
filtered_data['weekday'] = pd.to_datetime(filtered_data['date']).dt.day_name()
weekday_sales = filtered_data.groupby('weekday')['total_price'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
fig_weekly_sales = px.bar(weekday_sales, x='weekday',
                          y='total_price',
                          title='Sales by Weekday',
                          color='total_price',
                          color_continuous_scale=['#ffcccc', '#960018'],
                          labels={'weekday': '', 'total_price': 'revenue'}
                          )
fig_weekly_sales.update_coloraxes(showscale=False)

#st.markdown('##### Sales by Weekday')
#st.plotly_chart(fig_weekly_sales, use_container_width=True)

# Hourly Sales Trends
filtered_data['hour'] = pd.to_datetime(filtered_data['time'], format='%H:%M:%S').dt.hour
hourly_sales = filtered_data.groupby('hour')['total_price'].sum().reset_index()
fig_hourly_sales = px.bar(hourly_sales, x='hour',
                          y='total_price',
                          title='Hourly Sales Trend',
                          color='total_price',
                          color_continuous_scale=['#ffcccc', '#960018'],
                          labels={'total_price': 'Sales'}
                          )
fig_hourly_sales.update_coloraxes(showscale=False)


# Order size distribution
order_size = filtered_data.groupby('order_id')['quantity'].sum()
fig_order_size = px.histogram(order_size, nbins=20, title='Order size Distribution', color_discrete_sequence=['#960018'], labels={'count': '[No. of Orders]', 'value': 'Qty'})



# Price sensitivity
price_sensitivity = filtered_data.groupby('order_id').agg(
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



# Revenue Trends Over Time
filtered_data['month'] = filtered_data['date'].dt.to_period('M').dt.to_timestamp()
revenue_trend = filtered_data.groupby('month')['total_price'].sum().reset_index()
fig_revenue = px.line(revenue_trend, x='month', y='total_price',
                      title='Monthly Revenue Trend',
                      markers=True,
                      labels={'total_price': 'revenue'})
fig_revenue.update_traces(line_color='#960018', marker=dict(color='#960018'))


# Customer buying patterns
size_distribution = filtered_data.groupby('size')['quantity'].sum().reset_index()
fig_size = px.pie(size_distribution, values='quantity', names='size',
                  title='Pizza Size Preference',
                  color_discrete_sequence=['#960018'])



# Customer segmentation
customer_data = filtered_data.groupby('order_id')['total_price'].sum().reset_index()
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data)
fig_segment = px.scatter(customer_data, x=customer_data.index, 
                         y=customer_data['total_price'],
                         color=customer_data['cluster'],
                         title='Customer Segmentation based on Spend($)',
                         color_continuous_scale='magma',
                         labels={'total_price': 'Spend [$]'})



# Layout 3 column top-down story telling
#st.markdown('### 🔍 Sales Insights')

c1, c2, c3 = st.columns(3)

c1.plotly_chart(fig_category, use_container_width=True)
c2.plotly_chart(fig_top_pizzas, use_container_width=True)
c3.plotly_chart(fig_weekly_sales, use_container_width=True)


c4, c5, c6 = st.columns(3)
c4.plotly_chart(fig_hourly_sales, use_container_width=True)
c5.plotly_chart(fig_order_size, use_container_width=True)
c6.plotly_chart(fig_price_sensitivity, use_container_width=True)

c7, c8, c9 = st.columns(3)
c7.plotly_chart(fig_revenue, use_container_width=True)
c8.plotly_chart(fig_size, use_container_width=True)
c9.plotly_chart(fig_segment, use_container_width=True)


    # Footer
st.markdown("<hr>"
                "<center><em>DS/DA Ochieng Fess</em></center>"
                
                "<center>© 2025 Pizza Sales | Swift Traq "
                "</center>", unsafe_allow_html=True)