# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

# -------------------------
# 1️⃣ Read all sheets
# -------------------------
df = pd.read_excel("Flight_Data_AllSheets_Final.xlsx", sheet_name=None)
df = pd.concat(df.values(), ignore_index=True)

# -------------------------
# 2️⃣ Clean string columns
# -------------------------
str_cols = ["Flight Number","From","To","Aircraft","Runway","FlightType"]
for col in str_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).replace(r'\xa0', ' ', regex=True).str.strip()
        df[col] = df[col].replace('', np.nan)

# -------------------------
# 3️⃣ Forward-fill key columns
# -------------------------
for col in ["Flight Number","From","To","Aircraft"]:
    df[col] = df[col].ffill()

# -------------------------
# 4️⃣ Clean time columns
# -------------------------
def clean_time(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    match = re.search(r'(\d{1,2}:\d{2}:\d{2})', value)
    if match:
        return match.group(1)
    match = re.search(r'(\d{1,2}:\d{2}\s?[APap][Mm])', value)
    if match:
        return match.group(1)
    return None

time_cols = ["STD","ATD","STA","ATA"]
for col in time_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_time)
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

# -------------------------
# 5️⃣ Convert times to minutes
# -------------------------
def to_minutes(t):
    if pd.isna(t):
        return np.nan
    return t.hour*60 + t.minute + t.second/60

for col in time_cols:
    if col in df.columns:
        df[col+"_mins"] = df[col].apply(to_minutes)

# -------------------------
# 6️⃣ Calculate delays
# -------------------------
df["Duration"] = df["STA_mins"] - df["STD_mins"]
df["Departure_Delay"] = df["ATD_mins"] - df["STD_mins"]
df["ATA_mins"] = df["ATA_mins"].fillna(df["ATD_mins"] + df["Duration"])
df["Arrival_Delay"] = df["ATA_mins"] - df["STA_mins"]

# -------------------------
# 7️⃣ Runway & FlightType
# -------------------------
def infer_flight_type(to_airport):
    intl_airports = ["DXB","LHR","SIN","HKG","JFK"]
    if any(code in str(to_airport) for code in intl_airports):
        return "International"
    else:
        return "Domestic"

df["FlightType"] = df["To"].apply(infer_flight_type)

# Runway assignment per departure airport
runways = ["R1","R2","R3"]
airport_groups = df.groupby("From")

for airport, group in airport_groups:
    n = len(runways)
    for idx, row_idx in enumerate(group.index):
        df.at[row_idx, "Runway"] = runways[idx % n]

# -------------------------
# 8️⃣ Add congestion feature
# -------------------------
df['STD_hour'] = df['STD_mins'].apply(lambda x: int(x // 60) if pd.notna(x) else np.nan)
congestion = df.groupby('STD_hour').size()
df['Congestion'] = df['STD_hour'].map(congestion)

# -------------------------
# 9️⃣ Prepare multi-output features
# -------------------------
features = ["STD_mins","Duration","Departure_Delay","Congestion"]
df_model = df.dropna(subset=features + ["Arrival_Delay"])
# Target: Arrival Delay and Optimal_STD_mins (STD_mins adjusted by predicted delay)
df_model["Optimal_STD_mins"] = df_model["STD_mins"] - df_model["Arrival_Delay"]
X = df_model[features]
y = df_model[["Arrival_Delay","Optimal_STD_mins"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 1️⃣0️⃣ Train multi-output LightGBM
# -------------------------
lgb_model = MultiOutputRegressor(
    lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
)
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
print("MAE (Arrival Delay):", mean_absolute_error(y_test["Arrival_Delay"], y_pred[:,0]))
print("MAE (Optimal_STD_mins):", mean_absolute_error(y_test["Optimal_STD_mins"], y_pred[:,1]))
print("R² Score (Arrival Delay):", r2_score(y_test["Arrival_Delay"], y_pred[:,0]))
print("R² Score (Optimal_STD_mins):", r2_score(y_test["Optimal_STD_mins"], y_pred[:,1]))

# -------------------------
# 1️⃣1️⃣ Predict full dataset
# -------------------------
full_pred = lgb_model.predict(df[features].fillna(0))
df["Predicted_Delay"] = full_pred[:,0]
df["Optimal_STD_mins"] = full_pred[:,1]

def mins_to_time(m):
    if pd.isna(m):
        return np.nan
    h = int(m // 60) % 24
    m_int = int(m % 60)
    s = int((m - int(m)) * 60)
    return pd.Timestamp("2025-01-01") + pd.Timedelta(hours=h, minutes=m_int, seconds=s)

df["Optimal_STD"] = df["Optimal_STD_mins"].apply(mins_to_time).dt.time

# -------------------------
# 1️⃣2️⃣ Assign cascading delays
np.random.seed(42)
prob_cascade = 0.6
random_vals = np.random.exponential(scale=30, size=len(df))
random_vals = np.clip(random_vals, 0, 120)
cascade_flags = np.random.rand(len(df)) < prob_cascade
cascading_delay = np.where(cascade_flags, random_vals, 0.0)
df["CascadingDelay"] = cascading_delay
df["TotalPredictedDelay"] = df["Predicted_Delay"] + df["CascadingDelay"]

# -------------------------
# 1️⃣3️⃣ Save optimized schedule
final_schedule = df[[
    "Flight Number","From","To","Aircraft","STD","Optimal_STD",
    "Predicted_Delay","CascadingDelay","TotalPredictedDelay",
    "Runway","FlightType"
]].dropna(subset=["Flight Number","STD"])

final_schedule.to_csv("optimized_schedule_ai_multioutput.csv", index=False, encoding='utf-8-sig')
print("✅ Fully AI-driven optimized schedule saved as optimized_schedule_ai_multioutput.csv")

# -------------------------
# 1️⃣4️⃣ Export AI model
model_file = "flight_delay_model_ai_multioutput.pkl"
joblib.dump(lgb_model, model_file)
print(f"✅ Multi-output AI Model exported as {model_file}")
