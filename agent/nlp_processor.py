# nlp_processor.py

import re


class NLPProcessor:
    """
    Lớp này chịu trách nhiệm trích xuất thông tin từ câu hỏi của người dùng
    bằng cách sử dụng các biểu thức chính quy (regex) và phương pháp rule-based.
    """

    def __init__(self):
        """
        Khởi tạo NLPProcessor.
        """
        pass

    def extract_information(self, user_input: str) -> dict:
        """
        Trích xuất các thông số cần thiết từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Các thông số được trích xuất từ câu hỏi.
        """
        extracted_data = {
            'brand': self._extract_brand(user_input),
            'model': self._extract_model(user_input),
            'storage': self._extract_storage(user_input),
            'ram': self._extract_ram(user_input),
            'screen_size': self._extract_screen_size(user_input),
            'camera': self._extract_camera(user_input),
            'battery': self._extract_battery(user_input),
            'price': self._extract_price(user_input)
        }
        return extracted_data

    def _extract_brand(self, user_input: str) -> str:
        """
        Trích xuất thương hiệu điện thoại từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            str: Thương hiệu điện thoại (nếu tìm thấy).
        """
        brand_pattern = r"\b(?:Samsung|Apple|Xiaomi|OnePlus|Oppo|Realme|Vivo|Sony|Nokia|Huawei)\b"
        match = re.search(brand_pattern, user_input, re.IGNORECASE)
        return match.group(0) if match else None

    def _extract_model(self, user_input: str) -> str:
        """
        Trích xuất model của điện thoại từ câu hỏi.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            str: Model của điện thoại (nếu tìm thấy).
        """
        model_pattern = r"\b([A-Za-z0-9]+(?:\s[A-Za-z0-9]+)*)\b"
        match = re.search(model_pattern, user_input)
        return match.group(1) if match else None

    def _extract_storage(self, user_input: str) -> str:
        """
        Trích xuất thông số bộ nhớ (Storage) từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            str: Bộ nhớ của điện thoại (nếu tìm thấy).
        """
        storage_pattern = r"(\d+GB|\d+MB)"
        match = re.search(storage_pattern, user_input)
        return match.group(1) if match else None

    def _extract_ram(self, user_input: str) -> str:
        """
        Trích xuất thông số RAM từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            str: RAM của điện thoại (nếu tìm thấy).
        """
        ram_pattern = r"(\d+GB)\s*(?:RAM|ram)"
        match = re.search(ram_pattern, user_input)
        return match.group(1) if match else None

    def _extract_screen_size(self, user_input: str) -> float:
        """
        Trích xuất kích thước màn hình từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            float: Kích thước màn hình của điện thoại (nếu tìm thấy).
        """
        screen_pattern = r"(\d+(\.\d+)?)\s*(inch|inches)"
        match = re.search(screen_pattern, user_input)
        return float(match.group(1)) if match else None

    def _extract_camera(self, user_input: str) -> int:
        """
        Trích xuất thông số camera (MP) từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            int: Camera của điện thoại (nếu tìm thấy).
        """
        camera_pattern = r"(\d+)\s*(?:MP|megapixels|camera)"
        match = re.search(camera_pattern, user_input)
        return int(match.group(1)) if match else None

    def _extract_battery(self, user_input: str) -> int:
        """
        Trích xuất thông số pin (mAh) từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            int: Pin của điện thoại (nếu tìm thấy).
        """
        battery_pattern = r"(\d+)\s*(?:mAh|battery)"
        match = re.search(battery_pattern, user_input)
        return int(match.group(1)) if match else None

    def _extract_price(self, user_input: str) -> float:
        """
        Trích xuất giá của điện thoại từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            float: Giá của điện thoại (nếu tìm thấy).
        """
        price_pattern = r"(\d+)\s*(?:USD|\$|price)"
        match = re.search(price_pattern, user_input)
        return float(match.group(1)) if match else None
