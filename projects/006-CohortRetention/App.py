import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Cohort & Retention Dashboard", layout='centered',initial_sidebar_state='expanded')

base_dir = os.path.dirname(__file__)
data_folder = os.path.join(base_dir, "data")

data = pd.read_csv(os.path.join(data_folder, "UK_ecommerce_data.csv"), encoding='ISO-8859-1', dtype={'CustomerID': str})
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['CohortMonth'] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
data['TransactionMonth'] = data['InvoiceDate'].dt.to_period('M')
data['CohortIndex'] = (data['TransactionMonth'] - data['CohortMonth']).apply(lambda x: x.n)

data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
latest_date = data['InvoiceDate'].max()
customer_summary = data.groupby('CutsomerID').agg(
    TotalPurchases=('InvoiceNo', 'nunique'),
    TotalSpent=('TotalSpent', 'sum'),
    FirstPurchase=('InvoiceDate', 'min'),
    LastPurchase=('InvoiceDate', 'max')
).reset_index()
customer_summary['CustomerAge'] = (customer_summary['LastPurchase'] - customer_summary['FirstPurchase']).dt.days
customer_summary['Recency'] = (latest_date - customer_summary['LastPurchase']).dt.days
# define churn  15-months
customer_summary['Churn'] = (customer_summary['Recency'] > 450).astype(int)

# LOad model
best_model = joblib.load(os.path.join(data_folder, "best_cohort_model.pkl"))

# streamlit UI
st.title("Customer Cohort & Churn Analysis")

# Filters
st.sidebar.header("🔍 Filters")
date_range = st.sidebar.date_input("Select Date Range", [data['InvoiceDate'].min(), data['InvoiceDate'].max()])