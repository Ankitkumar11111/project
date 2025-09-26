import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

# --- 1. Load Data and Create Features (Same as before) ---
df = pd.read_csv("monthly_sales_2012_to_2022.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.month

# Add Season column (needed for encoding, even if not used in model)
def get_season(month):
    if month in [6,7,8,9]: return "Rainy"
    elif month in [10,11,12,1]: return "Winter"
    else: return "Summer"

df["Season"] = df["month"].apply(get_season)

# Encode categorical columns
le = LabelEncoder()
for col in ["Item_Name", "Item_Category", "Customer_Type", "Region", "Season"]:
    df[col] = le.fit_transform(df[col])

# Sort by Item & Date
df = df.sort_values(["Item_Name", "Date"])

# Create Lag and Rolling features (CRUCIAL 5 FEATURES)
df["lag_1"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(1)
df["lag_2"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(2)
df["roll_mean_3"] = df.groupby("Item_Name")["Monthly_Unit_Sales"].shift(1).rolling(3).mean()

# Drop rows with NaN (caused by lag/rolling)
df = df.dropna().reset_index(drop=True)


# --- 2. Feature Selection (NEW - Only 5 Features) ---
y = df["Monthly_Unit_Sales"]

# Selecting only the 5 chosen features:
X = df[[
    "Item_Name",    # What item? (Encoded)
    "month",        # When? (Time)
    "lag_1",        # Sales 1 month ago
    "lag_2",        # Sales 2 months ago
    "roll_mean_3"   # 3-month sales trend
]]


# --- 3. Model Training and Saving ---
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

# Save the NEW 5-feature model
joblib.dump(xgb_model, "xgb_model.pkl")

print("✅ New 5-Feature Model Trained and Saved to xgb_model.pkl")
print(f"Model trained on {X.shape[1]} features.")
