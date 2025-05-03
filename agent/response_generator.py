# response_generator.py

class ResponseGenerator:
    """
    Lớp này sinh phản hồi tự nhiên cho người dùng dựa trên kết quả dự đoán.
    Sau khi mô hình dự đoán điện thoại phù hợp, lớp này sẽ tạo câu trả lời dễ hiểu và thân thiện cho người dùng.
    """

    def __init__(self, phone_data):
        """
        Khởi tạo ResponseGenerator với dữ liệu điện thoại.

        Args:
            phone_data (pd.DataFrame): Dữ liệu thông số các mẫu điện thoại.
        """
        self.phone_data = phone_data

    def generate_response(self, predicted_phone):
        """
        Sinh câu trả lời cho người dùng dựa trên điện thoại dự đoán.

        Args:
            predicted_phone (str): Mô hình dự đoán điện thoại phù hợp.

        Returns:
            str: Câu trả lời tự nhiên cho người dùng.
        """
        # Tìm thông tin chi tiết về điện thoại đã được dự đoán
        phone_info = self.phone_data[self.phone_data['Model'] == predicted_phone].iloc[0]

        # Tạo câu trả lời dựa trên các thông số điện thoại
        response = (
            f"Chúng tôi đã tìm thấy một chiếc điện thoại phù hợp với yêu cầu của bạn: {predicted_phone}.\n"
            f"Thông số kỹ thuật của {predicted_phone} như sau:\n"
            f"- Bộ nhớ: {phone_info['Storage']} GB\n"
            f"- RAM: {phone_info['RAM']} GB\n"
            f"- Màn hình: {phone_info['Screen Size (inches)']} inch\n"
            f"- Camera: {phone_info['Camera (MP)']} MP\n"
            f"- Dung lượng pin: {phone_info['Battery Capacity (mAh)']} mAh\n"
            f"- Giá: ${phone_info['Price ($)']}\n\n"
            "Bạn có muốn nhận thêm thông tin hoặc tìm kiếm các mẫu điện thoại khác không?"
        )

        return response
