# Mô hình CNN 1D
# cnn_model_predictor.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Load & tiền xử lý ---
df = pd.read_csv("datas/cleaned_mobile_dataset (2).csv")
df.columns = [col.strip() for col in df.columns]
df.dropna(inplace=True)

for col in ['Brand', 'Model']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

numeric_cols = ['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)', 'Price ($)']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df[['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)']].values
y = df['Price ($)'].values

# --- Reshape dữ liệu cho CNN ---
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Mô hình CNN 1D ---
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

# --- Huấn luyện ---
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# --- Dự đoán ---
y_pred = model.predict(X_test).flatten()

# --- Đánh giá ---
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📊 MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"📈 R²: {r2_score(y_test, y_pred):.2f}")

# --- Lưu mô hình ---
os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")
print("✅ Đã lưu mô hình CNN vào model/cnn_model.h5")
