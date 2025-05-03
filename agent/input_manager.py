# input_manager.py

import re


class InputManager:
    """
    Lớp này chịu trách nhiệm xử lý câu hỏi đầu vào của người dùng,
    trích xuất các thông số như thương hiệu, model, bộ nhớ, RAM, màn hình, camera, pin, giá tiền.
    """

    def __init__(self):
        """
        Khởi tạo InputManager.
        """
        pass

    def process_input(self, user_input: str) -> dict:
        """
        Phân tích câu hỏi và trích xuất các thông số từ câu hỏi của người dùng,
        bao gồm thương hiệu, model, bộ nhớ, RAM, màn hình, camera, pin và giá tiền.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Các thông số (brand, model, storage, ram, screen_size, camera, battery_capacity, price)
                  được trích xuất từ câu hỏi.
        """
        specs = {
            'brand': None,
            'model': None,
            'storage': None,
            'ram': None,
            'screen_size': None,
            'camera': None,
            'battery_capacity': None,
            'price': None
        }

        specs.update(self._extract_brand_and_model(user_input))
        specs.update(self._extract_storage_and_ram(user_input))
        specs.update(self._extract_screen_and_camera(user_input))
        specs.update(self._extract_battery_and_price(user_input))

        return specs

    def _extract_brand_and_model(self, user_input: str) -> dict:
        """
        Trích xuất thông tin về thương hiệu và model từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Thông tin về thương hiệu và model.
        """
        brand_model = {}

        brand_patterns = [r"\b(?:Samsung|Apple|Xiaomi|OnePlus|Oppo|Realme)\b",
                          # Cập nhật danh sách thương hiệu theo yêu cầu
                          r"\b(?:Vivo|Sony|Nokia|Huawei)\b"]  # Thêm thương hiệu khác nếu cần

        model_pattern = r"\b([A-Za-z0-9]+(?:\s[A-Za-z0-9]+)*)\b"  # Tìm model kiểu 'Galaxy S21'

        for brand_pattern in brand_patterns:
            if re.search(brand_pattern, user_input, re.IGNORECASE):
                brand_model['brand'] = re.search(brand_pattern, user_input, re.IGNORECASE).group()

        model_match = re.search(model_pattern, user_input)
        if model_match:
            brand_model['model'] = model_match.group(1)

        return brand_model

    def _extract_storage_and_ram(self, user_input: str) -> dict:
        """
        Trích xuất thông số bộ nhớ trong (storage) và RAM từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Thông tin về bộ nhớ và RAM.
        """
        storage_ram = {}

        storage_match = re.search(r"(\d+GB|\d+MB)\s*(?:storage|bộ nhớ)", user_input, re.IGNORECASE)
        if storage_match:
            storage_ram['storage'] = storage_match.group(1)

        ram_match = re.search(r"(\d+GB)\s*(?:RAM|ram)", user_input, re.IGNORECASE)
        if ram_match:
            storage_ram['ram'] = ram_match.group(1)

        return storage_ram

    def _extract_screen_and_camera(self, user_input: str) -> dict:
        """
        Trích xuất thông số màn hình (screen size) và camera (megapixels) từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Thông tin về màn hình và camera.
        """
        screen_camera = {}

        screen_match = re.search(r"(\d+(\.\d+)?)\s*(inch|inches)", user_input, re.IGNORECASE)
        if screen_match:
            screen_camera['screen_size'] = screen_match.group(1)

        camera_match = re.search(r"(\d+)\s*(?:MP|megapixels|camera)", user_input, re.IGNORECASE)
        if camera_match:
            screen_camera['camera'] = camera_match.group(1)

        return screen_camera

    def _extract_battery_and_price(self, user_input: str) -> dict:
        """
        Trích xuất thông số pin (battery capacity) và giá tiền (price) từ câu hỏi của người dùng.

        Args:
            user_input (str): Câu hỏi của người dùng.

        Returns:
            dict: Thông tin về pin và giá tiền.
        """
        battery_price = {}

        battery_match = re.search(r"(\d+)\s*(?:mAh|battery)", user_input, re.IGNORECASE)
        if battery_match:
            battery_price['battery_capacity'] = battery_match.group(1)

        price_match = re.search(r"(\d+)\s*(?:USD|\$|price)", user_input, re.IGNORECASE)
        if price_match:
            battery_price['price'] = price_match.group(1)

        return battery_price
