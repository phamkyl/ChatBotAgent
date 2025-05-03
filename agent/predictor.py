# predictor.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.preprocess import DataPreprocessor


class PhonePredictor:
    """
    Lớp này sử dụng mô hình học máy đã huấn luyện để dự đoán điện thoại phù hợp
    với yêu cầu của người dùng dựa trên các đặc trưng của điện thoại.
    """

    def __init__(self, model_path: str, data_path: str):
        """
        Khởi tạo lớp PhonePredictor với đường dẫn mô hình và dữ liệu.

        Args:
            model_path (str): Đường dẫn đến mô hình đã huấn luyện.
            data_path (str): Đường dẫn đến dữ liệu điện thoại.
        """
        # Tải mô hình đã huấn luyện
        self.model = joblib.load(model_path)

        # Đọc dữ liệu điện thoại
        self.phone_data = pd.read_csv(data_path)

        # Tiền xử lý dữ liệu
        self.data_processor = DataPreprocessor(self.phone_data)
        self.processed_data = self.data_processor.process_data()

    def predict(self, user_input: dict) -> str:
        """
        Dự đoán điện thoại phù hợp dựa trên yêu cầu đầu vào từ người dùng.

        Args:
            user_input (dict): Một từ điển chứa yêu cầu của người dùng về thông số điện thoại.

        Returns:
            str: Dự đoán điện thoại phù hợp.
        """
        # Chuyển đổi yêu cầu người dùng thành DataFrame để xử lý
        input_df = pd.DataFrame([user_input])

        # Tiền xử lý dữ liệu đầu vào của người dùng
        input_processed = self.data_processor.process_data(input_df)

        # Dự đoán
        prediction = self.model.predict(input_processed)

        # Trả về tên điện thoại dự đoán từ dữ liệu đã đọc
        predicted_phone = self.phone_data.loc[prediction[0], 'Model']

        return predicted_phone

    def evaluate_model(self):
        """
        Đánh giá mô hình trên tập dữ liệu kiểm thử (nếu cần).
        """
        # Tách các đặc trưng và nhãn (Price)
        X = self.processed_data.drop(columns=['Brand', 'Model', 'Price ($)'])
        y = self.processed_data['Price ($)']

        # Dự đoán kết quả cho tập kiểm thử
        y_pred = self.model.predict(X)

        # Đánh giá độ chính xác của mô hình
        accuracy = accuracy_score(y, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
