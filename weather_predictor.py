import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('pune_may_2025_temperature.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Rename column for ease
df.rename(columns={'Avg Temp (°C)': 'Temperature'}, inplace=True)

# Sort by date
df = df.sort_values('Date')

# Add numeric day index for regression
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Prepare data for modeling
X = df[['Days']]
y = df['Temperature']

# Split dataset (no shuffle due to time series nature)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Predict for next 30 days
future_days = 30
last_day = df['Days'].max()
future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=future_days)
future_X = pd.DataFrame({'Days': range(last_day + 1, last_day + 1 + future_days)})
future_preds = model.predict(future_X)

# 📊 Create a combined figure
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 🔹 Subplot 1: Historical
axs[0].plot(df['Date'], df['Temperature'], label='Historical Temperature', marker='o')
axs[0].set_title('Historical Avg Temperature - Pune (May 2025)')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Temperature (°C)')
axs[0].grid(True)
axs[0].legend()
axs[0].tick_params(axis='x', rotation=45)

# 🔹 Subplot 2: Forecast
axs[1].plot(df['Date'], df['Temperature'], label='Historical', marker='o')
axs[1].plot(future_dates, future_preds, label='Predicted (Next 30 Days)', linestyle='--', marker='x')
axs[1].set_title('Forecast - Avg Temperature')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Temperature (°C)')
axs[1].grid(True)
axs[1].legend()
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle(f'Temperature Analysis & Forecast for Pune (RMSE: {rmse:.2f}°C)', fontsize=14, y=1.05)
plt.show()
