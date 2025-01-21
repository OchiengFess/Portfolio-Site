# Customer Lifetime Value Prediction

## Overview
This project estimates the lifetime value of customers using a combination of SQL and Python-based machine learning techniques. The analysis enables businesses to identify high-value customers and optimize marketing strategies.

## Workflow
1. **SQL Analysis**:
   - Data extraction and preprocessing.
   - Computation of historical metrics like total revenue and average transaction value.

2. **Python Model**:
   - Built a regression model to predict customer lifetime value.
   - Used libraries such as `scikit-learn` and `pandas`.

3. **Visualization**:
   - Presented results in an interactive Power BI dashboard.

## Files
- `clv_analysis.sql`: SQL script for data preprocessing and feature engineering.
- `clv_prediction.ipynb`: Jupyter Notebook for building the machine learning model.
- `sample_data.csv`: Dataset used in the analysis.
- `clv_dashboard.png`: Screenshot of the Power BI dashboard.

## Tools Used
- **SQL Server**
- **Python** (pandas, scikit-learn)
- **Power BI**

## Instructions
1. Import the `sample_data.csv` into your database.
2. Execute `clv_analysis.sql` to generate the necessary features.
3. Open `clv_prediction.ipynb` to run the model and view predictions.
