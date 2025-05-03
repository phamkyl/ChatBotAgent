from agent.input_manager import InputManager
from agent.nlp_processor import NLPProcessor
from agent.data_filter import DataFilter
from agent.predictor import Predictor
from agent.response_generator import ResponseGenerator
from utils.preprocess import DataPreprocessor  # Sửa tên import để sử dụng lớp DataPreprocessor
import pandas as pd
import gradio as gr


def load_data():
    """
    Hàm này dùng để tải dữ liệu từ các file CSV cần thiết (thông số điện thoại, câu hỏi ví dụ).
    """
    phone_specs = pd.read_csv('data/phone_specs.csv')

    # Tiến hành tiền xử lý dữ liệu bằng DataPreprocessor
    data_processor = DataPreprocessor(phone_specs)
    processed_data = data_processor.process_data()  # Tiến hành xử lý dữ liệu

    return processed_data


def initialize_agents(phone_specs):
    """
    Khởi tạo tất cả các thành phần trong hệ thống AI Agent.

    Args:
        phone_specs (pd.DataFrame): Dữ liệu thông số điện thoại.

    Returns:
        tuple: Trả về các đối tượng agent đã được khởi tạo.
    """
    # Khởi tạo các thành phần của hệ thống chatbot
    input_manager = InputManager()
    nlp_processor = NLPProcessor()
    data_filter = DataFilter(phone_specs)
    predictor = Predictor(phone_specs)
    response_generator = ResponseGenerator(phone_specs)

    return input_manager, nlp_processor, data_filter, predictor, response_generator


def chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor, response_generator):
    """
    Hàm xử lý câu hỏi của người dùng và sinh phản hồi từ chatbot.

    Args:
        user_input (str): Câu hỏi của người dùng.
        input_manager (InputManager): Quản lý đầu vào.
        nlp_processor (NLPProcessor): Xử lý ngôn ngữ tự nhiên.
        data_filter (DataFilter): Xử lý và chuẩn hóa dữ liệu điện thoại.
        predictor (Predictor): Dự đoán điện thoại phù hợp.
        response_generator (ResponseGenerator): Sinh phản hồi dựa trên kết quả dự đoán.

    Returns:
        str: Phản hồi từ chatbot.
    """
    # Xử lý đầu vào của người dùng
    preprocessed_input = input_manager.process_input(user_input)

    # Trích xuất thông tin từ câu hỏi người dùng
    extracted_info = nlp_processor.extract_info(preprocessed_input)

    # Lọc và chuẩn hóa dữ liệu từ thông số điện thoại
    filtered_data = data_filter.clean_and_validate_data(extracted_info)

    # Dự đoán điện thoại phù hợp
    predicted_phone = predictor.predict(filtered_data)

    # Sinh phản hồi cho người dùng dựa trên điện thoại dự đoán
    response = response_generator.generate_response(predicted_phone)

    return response


def create_gradio_interface(input_manager, nlp_processor, data_filter, predictor, response_generator):
    """
    Tạo giao diện chatbot với Gradio.

    Args:
        input_manager (InputManager): Quản lý đầu vào.
        nlp_processor (NLPProcessor): Xử lý ngôn ngữ tự nhiên.
        data_filter (DataFilter): Xử lý và chuẩn hóa dữ liệu điện thoại.
        predictor (Predictor): Dự đoán điện thoại phù hợp.
        response_generator (ResponseGenerator): Sinh phản hồi dựa trên kết quả dự đoán.

    Returns:
        Gradio.Interface: Giao diện Gradio.
    """
    # Tạo giao diện người dùng với Gradio
    chatbot_interface = gr.Interface(
        fn=lambda user_input: chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor,
                                            response_generator),
        inputs=gr.Textbox(lines=2, placeholder="Nhập câu hỏi của bạn..."),
        outputs=gr.Textbox(lines=5),
        title="Chatbot Tư Vấn Mua Điện Thoại",
        description="Chào bạn! Hãy nhập câu hỏi về điện thoại và tôi sẽ giúp bạn tìm sản phẩm phù hợp.",
    )

    return chatbot_interface


def main():
    """
    Hàm chính để khởi chạy ứng dụng.
    """
    # Tải dữ liệu điện thoại và tiền xử lý dữ liệu
    phone_specs = load_data()

    # Khởi tạo các agent
    input_manager, nlp_processor, data_filter, predictor, response_generator = initialize_agents(phone_specs)

    # Tạo giao diện chatbot với Gradio
    chatbot_interface = create_gradio_interface(input_manager, nlp_processor, data_filter, predictor,
                                                response_generator)

    # Chạy giao diện Gradio
    chatbot_interface.launch()


if __name__ == "__main__":
    main()
