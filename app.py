import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the Dataset
data_path = r'C:\Users\PAVPA\Downloads\archive_nifty\NIFTY50_all.csv'  # Update this path
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Volatility'] = data['High'] - data['Low']

# Sidebar Options
st.sidebar.header("Dashboard Settings")
selected_stocks = st.sidebar.multiselect(
    "Select Stocks", options=data['Symbol'].unique() if 'Symbol' in data.columns else []
)

# Filter Data Based on Selection
display_data = data[data['Symbol'].isin(selected_stocks)] if selected_stocks else data

# Dashboard Content
st.title("NIFTY 50 Stock Analysis")
st.header("Stock Data Overview")
st.dataframe(display_data.head())

# Volatility Plot
st.header("Volatility Trends")
fig, ax = plt.subplots(figsize=(12, 6))
if selected_stocks:
    for stock in selected_stocks:
        stock_data = data[data['Symbol'] == stock]
        ax.plot(stock_data['Date'], stock_data['Volatility'], label=stock)
else:
    ax.plot(data['Date'], data['Volatility'], label="All Stocks")
ax.legend()
ax.set_title("Volatility Over Time")
st.pyplot(fig)

# Heatmap
st.header("Feature Correlation Heatmap")
features = ['Open', 'High', 'Low', 'Close', 'Volume']
correlation_matrix = display_data[features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
