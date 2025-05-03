import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    Lớp này xử lý tiền xử lý dữ liệu như chuẩn hóa số liệu, mã hóa nhãn,
    và chuẩn hóa dữ liệu để đưa vào mô hình học máy.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Khởi tạo DataPreprocessor với dữ liệu đầu vào.

        Args:
            data (pd.DataFrame): Dữ liệu điện thoại.
        """
        self.data = data
        self.scaler = StandardScaler()  # Sử dụng StandardScaler để chuẩn hóa các cột số
        self.label_encoder = LabelEncoder()  # Mã hóa nhãn cho các cột 'Brand' và 'Model'

    def process_data(self) -> pd.DataFrame:
        """
        Tiến hành tiền xử lý dữ liệu, bao gồm chuẩn hóa và mã hóa.

        Returns:
            pd.DataFrame: Dữ liệu đã qua tiền xử lý.
        """
        # Kiểm tra và loại bỏ các giá trị null (nếu có)
        self.data = self.data.dropna()

        # Mã hóa nhãn thương hiệu và model
        self.data['Brand'] = self.label_encoder.fit_transform(self.data['Brand'])
        self.data['Model'] = self.label_encoder.fit_transform(self.data['Model'])

        # Chuẩn hóa các giá trị số (chỉ các cột số)
        numeric_columns = ['Storage ', 'RAM ', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)',
                           'Price ($)']

        # Kiểm tra nếu có giá trị âm trong các cột số, chúng ta có thể thay thế hoặc loại bỏ nếu cần
        for col in numeric_columns:
            self.data[col] = self.data[col].apply(
                lambda x: x if x > 0 else None)  # Thay giá trị âm bằng None (hoặc bạn có thể xử lý theo cách khác)

        self.data = self.data.dropna(subset=numeric_columns)  # Loại bỏ các dòng có giá trị null trong các cột số

        # Tiến hành chuẩn hóa
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])

        print("Dữ liệu đã được xử lý và chuẩn hóa.")
        return self.data
