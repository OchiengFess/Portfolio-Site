import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# Get base dir
base_dir = os.path.dirname(__file__)

# Build path to data folder
data_folder = os.path.join(base_dir, "data")

# Load data tables
def load_data(file_path, chunk_size=1000):
    """Loads sales, customer, and date data for cohort analysis."""
    df_sales = pd.concat(pd.read_csv(os.path.join(data_folder,"FactInternetSales.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_customer = pd.concat(pd.read_csv(os.path.join(data_folder,"DimCustomer.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)
    df_date = pd.concat(pd.read_csv(os.path.join(data_folder,"DimDate.csv"), chunksize=chunk_size, delimiter="\t"), ignore_index=True)

    df_sales['OrderDate'] = pd.to_datetime(df_sales['OrderDate'])
    df_date['FullDateAlternateKey'] = pd.to_datetime(df_date['FullDateAlternateKey'])

    df = pd.merge(df_sales, df_customer, on='CustomerKey', how='left')
    df = pd.merge(df, df_date, left_on='OrderDateKey', right_on='DateKey')

    return df

#prepare cohort data
def prepare_cohort_data(data):
    """Prepares cohort data with necessary calculations"""
    