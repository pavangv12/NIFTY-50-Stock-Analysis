import pandas as pd
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Dataset
path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")
csv_file_path = f"{path}/NIFTY50_all.csv"
data = pd.read_csv(csv_file_path)

# Task 1: Analyze the indicators in the dataset that best explain volatility and unpredictability
# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])  

# Check for missing values
if data.isnull().sum().sum() > 0:
    data.fillna(method='ffill', inplace=True)  

# Feature Engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['High'] - data['Low']
data['Moving_Avg_20'] = data['Close'].rolling(window=20).mean()
data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()

# Analyze Volatility Trends
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Volatility', data=data)
plt.title('Volatility Over Time')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()

# Task 2: Choose promising stocks based on analysis for the portfolio
if 'Symbol' in data.columns:
    stock_volatility = data.groupby('Symbol')['Volatility'].mean().sort_values()
    promising_stocks = stock_volatility.head()
    print("Promising stocks based on low volatility:", promising_stocks)
else:
    promising_stocks = None

# Task 3: Analyze performance of stocks
print("## Volatility Trends")
data[['Date', 'Volatility']].set_index('Date').plot(figsize=(12, 6))
plt.title("Volatility Trends")
plt.show()

if promising_stocks is not None:
    print("Promising Stocks:")
    print(promising_stocks)
else:
    print("Only one stock in the dataset.")

# Task 4: Feature engineer and build machine learning solutions
# Prepare data for ML model
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Moving_Avg_20', 'Moving_Avg_50']
data = data.dropna()  
X = data[features]
y = data['Close'].shift(-1).fillna(method='ffill')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE of Random Forest model: {rmse}")

# Feature Correlation Heatmap
correlation_matrix = data[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Close Price vs Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')
plt.legend()
plt.title("Close Price vs Predictions")
plt.show()