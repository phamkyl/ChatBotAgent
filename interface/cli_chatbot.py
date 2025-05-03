# interface/cli_chatbot.py
import sys
from agent.input_manager import InputManager
from agent.nlp_processor import NLPProcessor
from agent.data_filter import DataFilter
from agent.predictor import PhonePredictor
from agent.response_generator import ResponseGenerator
from utils.preprocess import DataPreprocessor
import pandas as pd


def load_data():
    """
    Hàm này dùng để tải dữ liệu từ các file CSV cần thiết (thông số điện thoại, câu hỏi ví dụ).
    """
    phone_specs = pd.read_csv('D:/project_DACN3/ChatBotAgent/datas/cleaned_mobile_dataset (2).csv')

    # Tiến hành tiền xử lý dữ liệu bằng DataPreprocessor
    data_processor = DataPreprocessor(phone_specs)
    processed_data = data_processor.process_data()  # Tiến hành xử lý dữ liệu

    return processed_data


def initialize_agents():
    """
    Khởi tạo tất cả các thành phần trong hệ thống AI Agent.
    """
    input_manager = InputManager()
    nlp_processor = NLPProcessor()

    # Cập nhật đường dẫn file CSV
    phone_specs_file = "D:/project_DACN3/ChatBotAgent/datas/cleaned_mobile_dataset (2).csv"

    # Khởi tạo DataFilter với đường dẫn file CSV
    data_filter = DataFilter(phone_specs_file)

    # Cập nhật đường dẫn mô hình và dữ liệu
    model_path = "D:/project_DACN3/ChatBotAgent/model/phone_model.pkl"
    predictor = PhonePredictor(model_path, phone_specs_file)
    response_generator = ResponseGenerator(phone_specs_file)

    return input_manager, nlp_processor, data_filter, predictor, response_generator


def chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor, response_generator):
    """
    Hàm xử lý câu hỏi của người dùng và sinh phản hồi từ chatbot.
    """
    preprocessed_input = input_manager.process_input(user_input)
    extracted_info = nlp_processor.extract_information(preprocessed_input)
    filtered_data = data_filter.clean_and_validate_data(extracted_info)

    # Dự đoán điện thoại dựa trên thông tin đầu vào người dùng
    predicted_phone = predictor.predict(filtered_data)

    # Sinh phản hồi từ chatbot
    response = response_generator.generate_response(predicted_phone)
    return response


def main():
    """
    Hàm chính để chạy chatbot trên giao diện dòng lệnh.
    """
    phone_specs = load_data()  # Dữ liệu đã được xử lý
    # Gọi hàm initialize_agents mà không truyền tham số
    input_manager, nlp_processor, data_filter, predictor, response_generator = initialize_agents()

    print("📱 Chatbot Tư Vấn Mua Điện Thoại - Giao Diện Dòng Lệnh")
    print("Nhập 'exit' để thoát.\n")

    while True:
        user_input = input("👤 Bạn: ")

        if user_input.lower() == "exit":
            print("🤖 Chatbot: Hẹn gặp lại bạn!")
            break

        # Gọi hàm xử lý câu hỏi và sinh phản hồi
        response = chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor, response_generator)

        print(f"🤖 Chatbot: {response}")



if __name__ == "__main__":
    main()
