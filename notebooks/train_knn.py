# train_knn.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# --- Đọc dữ liệu ---
df = pd.read_csv('../datas/cleaned_mobile_dataset (2).csv')

# --- Tiền xử lý dữ liệu ---
# Xử lý tên cột
df.columns = [col.strip() for col in df.columns]

# Xóa các hàng thiếu dữ liệu
df = df.dropna()

# Mã hóa nhãn (Brand, Model)
for col in ['Brand', 'Model']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Chuẩn hóa các cột số
scaler = StandardScaler()
numeric_cols = ['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)', 'Price ($)']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Tách dữ liệu thành X, y
X = df.drop(columns=['Brand', 'Model', 'Price ($)'])
y = df['Price ($)']

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Huấn luyện mô hình KNN ---
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# --- Dự đoán ---
y_pred = model.predict(X_test)

# --- Đánh giá ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.2f}")
print(f"📊 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# --- Lưu mô hình ---
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/knn_model.pkl')
print("✅ Mô hình KNN đã được lưu vào 'model/knn_model.pkl'")
