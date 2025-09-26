import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("monthly_sales_2012_to_2022.csv")

# --- Your preprocessing logic (from your old app.py) ---
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['year'] = df['Date'].dt.year

# Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['Season'] = df['month'].apply(get_season)

# Lag and rolling mean
df['lag_1'] = df['Monthly_Sales'].shift(1)
df['lag_2'] = df['Monthly_Sales'].shift(2)
df['roll_mean_3'] = df['Monthly_Sales'].rolling(3).mean()
df['roll_mean_6'] = df['Monthly_Sales'].rolling(6).mean()
df = df.dropna()

# Encode categorical
categorical_cols = ['Item_Name', 'Item_Category', 'Customer_Type', 'Region', 'Season']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Features & Target
X = df.drop(columns=['Date', 'Monthly_Sales'])
y = df['Monthly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# Save model
joblib.dump(xgb, "xgb_model.pkl")
print("✅ Model trained and saved as xgb_model.pkl")
