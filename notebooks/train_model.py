# Mô hình RandomForestRegressor
# Import các thư viện cần thiết
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import sys
sys.path.append("..")  # Thêm thư mục cha vào đường dẫn

# --- Đọc dữ liệu (dùng raw string để tránh lỗi escape sequence) ---
phone_data = pd.read_csv(r'/datas/cleaned_mobile_dataset (2).csv')

# --- Tiền xử lý dữ liệu ---
class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def process_data(self) -> pd.DataFrame:
        # Xử lý khoảng trắng trong tên cột
        self.data.columns = [col.strip() for col in self.data.columns]

        # Loại bỏ giá trị thiếu
        self.data = self.data.dropna()

        # Mã hóa nhãn (sử dụng encoder riêng cho từng cột)
        for col in ['Brand', 'Model']:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le  # lưu nếu cần dùng sau

        # Chuẩn hóa các cột số
        numeric_cols = ['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)', 'Price ($)']
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

        print("✅ Dữ liệu đã được xử lý và chuẩn hóa.")
        return self.data

# Khởi tạo và xử lý dữ liệu
data_processor = DataPreprocessor(phone_data)
processed_data = data_processor.process_data()

# Tách X và y
X = processed_data.drop(columns=['Brand', 'Model', 'Price ($)'])
y = processed_data['Price ($)']

# Tách tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.2f}")
print(f"📊 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# --- Lưu mô hình ---
os.makedirs('model', exist_ok=True)  # Tạo thư mục nếu chưa có
joblib.dump(model, '../model/phone_model.pkl')
print("✅ Mô hình đã được lưu vào 'model/phone_model.pkl'")
