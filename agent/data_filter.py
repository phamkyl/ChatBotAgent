# data_filter.py

import pandas as pd


class DataFilter:
    """
    Lớp này chịu trách nhiệm làm sạch, chuẩn hóa và validate dữ liệu đầu vào,
    giúp cải thiện chất lượng và tính chính xác của dữ liệu trước khi đưa vào mô hình.
    """

    def __init__(self, phone_specs_file: str):
        """
        Khởi tạo DataFilter với dữ liệu điện thoại.

        Args:
            phone_specs_file (str): Đường dẫn tới file dữ liệu điện thoại (CSV).
        """
        self.phone_specs = pd.read_csv(phone_specs_file)

    def clean_data(self):
        """
        Làm sạch dữ liệu: Loại bỏ các giá trị thiếu và chuẩn hóa dữ liệu.
        """
        # Loại bỏ các dòng có giá trị thiếu
        self.phone_specs.dropna(
            subset=['Brand', 'Model', 'Storage', 'RAM', 'Screen Size (inches)', 'Camera (MP)', 'Battery Capacity (mAh)',
                    'Price ($)'], inplace=True)

        # Chuẩn hóa dữ liệu (chuyển tất cả về chữ thường để so sánh dễ dàng hơn)
        self.phone_specs['Brand'] = self.phone_specs['Brand'].str.lower()
        self.phone_specs['Model'] = self.phone_specs['Model'].str.lower()
        self.phone_specs['Storage'] = self.phone_specs['Storage'].str.lower()
        self.phone_specs['RAM'] = self.phone_specs['RAM'].str.lower()
        self.phone_specs['Screen Size (inches)'] = self.phone_specs['Screen Size (inches)'].astype(float)
        self.phone_specs['Camera (MP)'] = self.phone_specs['Camera (MP)'].astype(float)
        self.phone_specs['Battery Capacity (mAh)'] = self.phone_specs['Battery Capacity (mAh)'].astype(float)
        self.phone_specs['Price ($)'] = self.phone_specs['Price ($)'].astype(float)

        print("Dữ liệu đã được làm sạch và chuẩn hóa.")
        return self.phone_specs

    def validate_data(self):
        """
        Validate các thông số dữ liệu để đảm bảo tính chính xác và hợp lệ.
        """
        # Kiểm tra dữ liệu bộ nhớ và RAM có phải là số và lớn hơn 0 không
        invalid_storage = self.phone_specs[~self.phone_specs['Storage'].str.contains(r'\d+GB|\d+MB')]
        invalid_ram = self.phone_specs[~self.phone_specs['RAM'].str.contains(r'\d+GB')]

        # Kiểm tra giá trị pin và camera không phải NaN hoặc 0
        invalid_battery = self.phone_specs[self.phone_specs['Battery Capacity (mAh)'] <= 0]
        invalid_camera = self.phone_specs[self.phone_specs['Camera (MP)'] <= 0]

        if not invalid_storage.empty:
            print(f"Những dòng có bộ nhớ không hợp lệ:\n{invalid_storage}")
        if not invalid_ram.empty:
            print(f"Những dòng có RAM không hợp lệ:\n{invalid_ram}")
        if not invalid_battery.empty:
            print(f"Những dòng có pin không hợp lệ:\n{invalid_battery}")
        if not invalid_camera.empty:
            print(f"Những dòng có camera không hợp lệ:\n{invalid_camera}")

        return invalid_storage.empty and invalid_ram.empty and invalid_battery.empty and invalid_camera.empty
