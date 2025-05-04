# train_dnn_model.py

import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Đọc dữ liệu ---
df = pd.read_csv('datas/cleaned_mobile_dataset (2).csv')

# --- Tiền xử lý dữ liệu ---
df.columns = [col.strip() for col in df.columns]
df = df.dropna()

# Mã hóa nhãn
for col in ['Brand', 'Model']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
numeric_cols = ['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)', 'Price ($)']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Tách X và y
X = df.drop(columns=['Brand', 'Model', 'Price ($)'])
y = df['Price ($)']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Khởi tạo mô hình DNN ---
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Huấn luyện mô hình ---
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# --- Dự đoán & Đánh giá ---
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.2f}")
print(f"📊 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# --- Lưu mô hình ---
os.makedirs('model', exist_ok=True)
model.save('model/dnn_model.h5')
print("✅ Mô hình DNN đã được lưu vào 'model/dnn_model.h5'")
