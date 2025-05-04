import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Load dữ liệu
df = pd.read_csv('../datas/cleaned_mobile_dataset (2).csv')

# Tiền xử lý
df.columns = [col.strip() for col in df.columns]
df = df.dropna()

for col in ['Brand', 'Model']:
    df[col] = LabelEncoder().fit_transform(df[col])

scaler = StandardScaler()
numeric_cols = ['Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)', 'Price ($)']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop(columns=['Brand', 'Model', 'Price ($)'])
y = df['Price ($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Lưu model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/decision_tree_model.pkl')
