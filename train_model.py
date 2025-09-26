import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import xgboost as xgb

# Load dataset
df = pd.read_csv("monthly_sales_2012_to_2022.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract time-based features
df["month"] = df["Date"].dt.month
df["quarter"] = df["Date"].dt.quarter
df["year"] = df["Date"].dt.year

# Add Season column
def get_season(month):
    if month in [6,7,8,9]:   # June–Sep → Rainy
        return "Rainy"
    elif month in [10,11,12,1]:  # Oct–Jan → Winter
        return "Winter"
    else:   # Feb–May → Summer
        return "Summer"

df["Season"] = df["month"].apply(get_season)

# Encode categorical columns
le = LabelEncoder()
for col in ["Item_Name", "Item_Category", "Customer_Type", "Region", "Season"]:
    df[col] = le.fit_transform(df[col])


# Sort by Item & Date
df = df.sort_values(["Item_Name", "Date"])

# Create lag features (previous months’ sales)
df["lag_1"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(1)
df["lag_2"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(2)

# Rolling mean features
df["roll_mean_3"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(1).rolling(3).mean()
df["roll_mean_6"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(1).rolling(6).mean()

# Drop rows with NaN (caused by lag/rolling)
df = df.dropna().reset_index(drop=True)

# Dependent variable (target)
y = df["Monthly_Unit_Sales"]

# Independent variables (all predictors)
X = df.drop(["Monthly_Unit_Sales", "Date"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Remove 'Predicted' and 'Actual' columns from X_test if they exist
if 'Predicted' in X_test.columns:
    X_test = X_test.drop('Predicted', axis=1)
if 'Actual' in X_test.columns:
    X_test = X_test.drop('Actual', axis=1)

y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 XGBoost Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Group predictions by season
X_test["Predicted"] = y_pred
X_test["Actual"] = y_test.values

season_summary = X_test.groupby("Season")[["Predicted", "Actual"]].sum()
print("\n=== Seasonal Predicted vs Actual Sales ===")
print(season_summary)

feature_cols = [
    'Item_Name', 'Item_Category', 'Customer_Type', 'Region',
    'Monthly_Sales', 'Avg_Unit_Price', 'Avg_Discount',
    'month', 'quarter', 'year', 'Season',
    'lag_1', 'lag_2', 'roll_mean_3', 'roll_mean_6'
]

import pandas as pd

# Load your dataset
df = pd.read_csv("monthly_sales_2012_to_2022.csv")

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract year and month
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Define rainy season months (June-Sep)
rainy_months = [6, 7, 8, 9]

# Filter for 2022 rainy season
df_2022_rainy = df[(df["year"] == 2022) & (df["month"].isin(rainy_months))]

# Group by Item_Name and sum sales
rainy_sales_2022 = df_2022_rainy.groupby("Item_Name")[["Monthly_Sales", "Monthly_Unit_Sales"]].sum().reset_index()

# Assume 2023 prediction = same as 2022 (or you can apply growth factor, e.g., *1.05 for 5% growth)
rainy_sales_2022["Predicted_2023_Unit_Sales"] = (rainy_sales_2022["Monthly_Unit_Sales"] * 1.05).astype(int)

# Sort by predicted sales (descending)
rainy_sales_2023 = rainy_sales_2022.sort_values(by="Predicted_2023_Unit_Sales", ascending=False)

# Print results
print("📌 Predicted Items for Rainy Season 2023 (using 2022 data):")
for i, row in rainy_sales_2023.iterrows():
    print(f"   {row['Item_Name']} - {row['Predicted_2023_Unit_Sales']}")

import pandas as pd

# Load your dataset
df = pd.read_csv("monthly_sales_2012_to_2022.csv")

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract year and month
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Define winter season months (Dec, Jan, Feb)
winter_months = [12, 1, 2]

# Filter for 2022 winter season
df_2022_winter = df[(df["year"] == 2022) & (df["month"].isin(winter_months))]

# Group by Item_Name and sum sales
winter_sales_2022 = df_2022_winter.groupby("Item_Name")[["Monthly_Sales", "Monthly_Unit_Sales"]].sum().reset_index()

# Assume 2023 prediction = same as 2022 (or with growth factor, e.g., *1.05 for 5% growth)
winter_sales_2022["Predicted_2023_Unit_Sales"] = (winter_sales_2022["Monthly_Unit_Sales"] * 1.05).astype(int)

# Sort by predicted sales (descending)
winter_sales_2023 = winter_sales_2022.sort_values(by="Predicted_2023_Unit_Sales", ascending=False)

# Print results
print("📌 Predicted Items for Winter Season 2023 (using 2022 data):")
for i, row in winter_sales_2023.iterrows():
    print(f"   {row['Item_Name']} - {row['Predicted_2023_Unit_Sales']}")

import pandas as pd

# Load your dataset
df = pd.read_csv("monthly_sales_2012_to_2022.csv")

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract year and month
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Define summer season months (March, April, May)
summer_months = [3, 4, 5]

# Filter for 2022 summer season
df_2022_summer = df[(df["year"] == 2022) & (df["month"].isin(summer_months))]

# Group by Item_Name and sum sales
summer_sales_2022 = df_2022_summer.groupby("Item_Name")[["Monthly_Sales", "Monthly_Unit_Sales"]].sum().reset_index()

# Assume 2023 prediction = same as 2022 (or with growth factor, e.g., *1.05 for 5% growth)
summer_sales_2022["Predicted_2023_Unit_Sales"] = (summer_sales_2022["Monthly_Unit_Sales"] * 1.05).astype(int)

# Sort by predicted sales (descending)
summer_sales_2023 = summer_sales_2022.sort_values(by="Predicted_2023_Unit_Sales", ascending=False)

# Print results
print("📌 Predicted Items for Summer Season 2023 (using 2022 data):")
for i, row in summer_sales_2023.iterrows():
    print(f"   {row['Item_Name']} - {row['Predicted_2023_Unit_Sales']}")


import joblib
joblib.dump(xgb_model, "xgb_model.pkl")

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("xgb_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

@app.route('/')
def home():
    return "✅ ML API is running! Use POST /predict with JSON data."

if __name__ == '__main__':
    app.run(debug=True, port=5000)
