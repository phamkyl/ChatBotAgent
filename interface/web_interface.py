# interface/web_interface.py
import gradio as gr
from agent.input_manager import InputManager
from agent.nlp_processor import NLPProcessor
from agent.data_filter import DataFilter
from agent.predictor import Predictor
from agent.response_generator import ResponseGenerator
from interface.cli_chatbot import initialize_agents
from utils.preprocess import DataPreprocessor
import pandas as pd


def load_data():
    """
    Hàm này dùng để tải dữ liệu từ các file CSV cần thiết (thông số điện thoại, câu hỏi ví dụ).
    """
    phone_specs = pd.read_csv('data/phone_specs.csv')

    # Tiến hành tiền xử lý dữ liệu bằng DataPreprocessor
    data_processor = DataPreprocessor(phone_specs)
    processed_data = data_processor.process_data()  # Tiến hành xử lý dữ liệu

    return processed_data


def create_gradio_interface(input_manager, nlp_processor, data_filter, predictor, response_generator):
    """
    Tạo giao diện chatbot với Gradio.
    """
    chatbot_interface = gr.Interface(
        fn=lambda user_input: chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor,
                                            response_generator),
        inputs=gr.Textbox(lines=2, placeholder="Nhập câu hỏi của bạn..."),
        outputs=gr.Textbox(lines=5),
        title="Chatbot Tư Vấn Mua Điện Thoại",
        description="Chào bạn! Hãy nhập câu hỏi về điện thoại và tôi sẽ giúp bạn tìm sản phẩm phù hợp.",
    )

    return chatbot_interface


def chat_with_bot(user_input, input_manager, nlp_processor, data_filter, predictor, response_generator):
    """
    Hàm xử lý câu hỏi của người dùng và sinh phản hồi từ chatbot.
    """
    preprocessed_input = input_manager.process_input(user_input)
    extracted_info = nlp_processor.extract_info(preprocessed_input)
    filtered_data = data_filter.clean_and_validate_data(extracted_info)
    predicted_phone = predictor.predict(filtered_data)
    response = response_generator.generate_response(predicted_phone)
    return response


def main():
    """
    Hàm chính để chạy ứng dụng Gradio.
    """
    phone_specs = load_data()
    input_manager, nlp_processor, data_filter, predictor, response_generator = initialize_agents(phone_specs)

    chatbot_interface = create_gradio_interface(input_manager, nlp_processor, data_filter, predictor,
                                                response_generator)

    chatbot_interface.launch()


if __name__ == "__main__":
    main()
