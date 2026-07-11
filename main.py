import pandas as pd

# Load dataset
df = pd.read_csv("sales.csv")

# Show first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
import matplotlib.pyplot as plt

# Convert Date column into datetime
df["Date"] = pd.to_datetime(df["Date"])

# Plot Sales Trend
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["Sales"], marker='o')
plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np

# Create Day Number (1,2,3...)
df["Day"] = np.arange(1, len(df) + 1)

# Features and Target
X = df[["Day"]]
y = df["Sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict Sales
df["Predicted_Sales"] = model.predict(X)

print("\nPredicted Sales:")
print(df[["Day", "Sales", "Predicted_Sales"]])
# Actual vs Predicted Graph

plt.figure(figsize=(10,5))

plt.plot(df["Day"], df["Sales"], marker="o", label="Actual Sales")
plt.plot(df["Day"], df["Predicted_Sales"], linestyle="--", label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y, df["Predicted_Sales"])
rmse = np.sqrt(mean_squared_error(y, df["Predicted_Sales"]))
r2 = r2_score(y, df["Predicted_Sales"])

print("\nModel Accuracy")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,2))
# Predict Sales for Next 7 Days

future_days = np.array([[31], [32], [33], [34], [35], [36], [37]])

future_predictions = model.predict(future_days)

print("\nNext 7 Days Sales Prediction:")

for day, sale in zip(future_days.flatten(), future_predictions):
    print(f"Day {day}: {sale:.2f}")
    plt.figure(figsize=(10,5))

# Historical Sales
plt.plot(df["Day"], df["Sales"], marker="o", label="Historical Sales")

# Future Prediction
plt.plot(future_days, future_predictions, marker="o", linestyle="--", color="red", label="Future Prediction")

plt.title("Sales Forecast for Next 7 Days")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.show()
