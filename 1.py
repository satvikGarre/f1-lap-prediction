import fastf1
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

os.makedirs('./fastf1_cache', exist_ok=True)
fastf1.Cache.enable_cache('./fastf1_cache')

def get_lap_features(year, gp, session_type, driver_code):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps = session.laps.pick_drivers([driver_code]).sort_values('LapNumber')

    # Basic features
    features = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                'Compound', 'LapNumber', 'TyreLife', 'PitOutTime', 'PitInTime', 'TrackStatus']
    laps = laps[features]

    # Drop rows missing key times
    laps = laps.dropna(subset=['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time'])

    # Convert time columns to seconds
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        laps[col] = pd.to_timedelta(laps[col]).dt.total_seconds()

    # Encode categorical
    if 'Compound' in laps.columns:
        laps['Compound'] = laps['Compound'].astype('category').cat.codes

    # Tyre age
    laps['TyreAge'] = laps['TyreLife'].fillna(method='ffill').fillna(0)

    # Pit lap indicator
    laps['IsPitLap'] = laps[['PitInTime', 'PitOutTime']].notna().any(axis=1).astype(int)

    # Track status (safety car, etc.)
    if 'TrackStatus' in laps.columns:
        laps['TrackStatus'] = laps['TrackStatus'].astype('category').cat.codes

    # Drop unused columns
    laps = laps.drop(['TyreLife', 'PitInTime', 'PitOutTime'], axis=1)

    return laps

data_frames = []
for (year, gp, session_type) in [(2025, 'Monza', 'R'), (2024, 'Monza', 'R'), (2023, 'Monza', 'R'), (2022, 'Monza', 'R'),(2021, 'Monza', 'R'), (2020, 'Monza', 'R'),]:
    try:
        df = get_lap_features(year, gp, session_type, 'VER')
        print(f"Loaded {len(df)} rows for {year} {gp} {session_type}")
        data_frames.append(df)
    except Exception as e:
        print(f"Failed to load {year} {gp} {session_type}: {e}")

full_df = pd.concat(data_frames, ignore_index=True)
print(f"Combined data shape: {full_df.shape}")
print(full_df.head())

# -------------------- Preprocessing --------------------
target = 'LapTime'
feature_cols = [col for col in full_df.columns if col != target]

X = full_df[feature_cols].values
y = full_df[target].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

sequence_length = 10
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y_scaled[i+sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"Sequenced data shape: X={X_seq.shape}, y={y_seq.shape}")

# Train/test split
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# -------------------- LSTM Model --------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=200, batch_size=16,
                    validation_data=(X_test, y_test), verbose=0)

# Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test)

# -------------------- Random Forest Baseline --------------------
X_rf = X_seq.reshape(X_seq.shape[0], -1)
X_train_rf, X_test_rf = X_rf[:split_idx], X_rf[split_idx:]
y_train_rf, y_test_rf = y_seq[:split_idx], y_seq[split_idx:]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_rf, y_train_rf.ravel())
y_pred_rf = scaler_y.inverse_transform(rf.predict(X_test_rf).reshape(-1, 1))

# -------------------- Gradient Boosting Baseline --------------------
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
gb.fit(X_train_rf, y_train_rf.ravel())
y_pred_gb = scaler_y.inverse_transform(gb.predict(X_test_rf).reshape(-1, 1))

# -------------------- Metrics --------------------
def print_metrics(true, pred, model_name):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}\n")

print_metrics(y_test_inv, y_pred, "LSTM Model")
print_metrics(y_test_inv, y_pred_rf, "Random Forest")
print_metrics(y_test_inv, y_pred_gb, "Gradient Boosting")

# -------------------- Plot --------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual Lap Time (s)')
plt.plot(y_pred, label='LSTM Predicted', alpha=0.8)
plt.plot(y_pred_rf, label='Random Forest Predicted', linestyle='--')
plt.plot(y_pred_gb, label='Gradient Boosting Predicted', linestyle='-.')
plt.title('Lap Time Prediction - LSTM vs RF vs GBM')
plt.xlabel('Lap Index (Test Set)')
plt.ylabel('Lap Time (seconds)')
plt.legend()
plt.show()
