cây thư mục 
chatbot_phone_recommender/
│
├── data/                             # Dữ liệu đầu vào và đầu ra
│   ├── phone_specs.csv              # Dataset chứa thông số điện thoại
│   └── example_questions.txt        # Một số câu hỏi ví dụ
│
├── agent/                            # Các thành phần AI Agent
│   ├── input_manager.py             # Xử lý câu hỏi đầu vào
│   ├── nlp_processor.py             # Trích xuất thông tin bằng regex, rule-based
│   ├── data_filter.py               # Làm sạch, chuẩn hóa và validate dữ liệu
│   ├── predictor.py                 # Mô hình ML dự đoán điện thoại phù hợp
│   └── response_generator.py        # Sinh phản hồi dựa trên kết quả dự đoán
│
├── model/                            # Lưu mô hình đã huấn luyện
│   └── phone_model.pkl              # Mô hình RandomForest hoặc KNN đã huấn luyện
│
├── utils/                            # Hàm hỗ trợ chung
│   └── preprocess.py                # Tiền xử lý dữ liệu, encode, chuẩn hóa
│
├── interface/                        # Giao diện người dùng
│   ├── cli_chatbot.py               # Chatbot chạy trên giao diện dòng lệnh
│   └── web_interface.py             # Giao diện web (Gradio, Streamlit)
│
├── notebooks/                        # Notebook để thử nghiệm
│   └── train_model.ipynb            # Notebook huấn luyện mô hình
│
├── main.py                           # File chính để chạy ứng dụng
├── requirements.txt                  # Thư viện cần thiết
└── README.md                         # Tài liệu mô tả dự án

Các mô hình huấn luyện dự đoán điện thoại :  
- RandomForestRegressor ,
- Decision Tree, 
- Neural Network (DNN),
- Mô hình K-Nearest Neighbors (KNN), 
- Linear Regression
các mô hình huấn luện tự sinh câu trả lời tự nhiên :
- Seq2Seq (Sequence-to-Sequence),
- T5 (Text-to-Text Transfer Transformer),
- GPT,
- BART 
data : dữ liệu với 
- thông tin cấu hình
- thông tin câu hỏi
Ai Agent - lý thuyết
Tiền xử lí dữ liệu
Đánh giá các mô hình