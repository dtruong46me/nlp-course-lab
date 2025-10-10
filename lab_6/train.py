# train.py
import os
import pandas as pd
from typing import Literal
import pickle  # Thay thế joblib bằng pickle

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

__root__ = os.getcwd()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Thực hiện các bước tiền xử lý dữ liệu email."""
    print("Bắt đầu tiền xử lý dữ liệu...")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['text'] = df['text'].astype(str) # type: ignore
    df['text'] = df['text'].str.replace(r'[\n\r\t]', ' ', regex=True) # type: ignore
    df['text'] = df['text'].str.replace(r'^Subject: ', '', regex=True, n=1) # type: ignore
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    print("Tiền xử lý hoàn tất.")
    return df

def run_training(
    vectorizer_type: Literal['count', 'hash'] = 'count',
    model_type: Literal['naive_bayes', 'logistic'] = 'naive_bayes'
):
    """
    Hàm chính để thực hiện toàn bộ quy trình huấn luyện.

    Args:
        vectorizer_type (Literal): Loại vectorizer ('count' hoặc 'hash').
        model_type (Literal): Loại mô hình ('naive_bayes' hoặc 'logistic').
    """
    # ---- 1. Tải và tiền xử lý dữ liệu ----
    data_path = os.path.join(__root__, "lab_6", "spam_data", "spam_ham_dataset.csv")
    # Giả định bạn đã tạo thư mục lab_6, nếu không có thể bỏ đi
    # data_path = os.path.join(__root__, "lab_6", "spam_data", "spam_ham_dataset.csv")

    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{data_path}'")
        print("Vui lòng tải và đặt file vào đúng thư mục.")
        return

    data = pd.read_csv(data_path)
    processed_data = preprocess_data(data)

    X = processed_data['text']
    y = processed_data['label_num']
    
    # ---- 2. Lựa chọn và huấn luyện Vectorizer ----
    print(f"Sử dụng {vectorizer_type.capitalize()}Vectorizer...")
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words='english', max_df=0.9)
    else: # hash
        vectorizer = HashingVectorizer(stop_words='english', n_features=2**14, norm='l2')

    X_vec = vectorizer.fit_transform(X)

    # ---- 3. Lựa chọn và huấn luyện Model ----
    print(f"Huấn luyện mô hình {model_type.replace('_', ' ').title()}...")
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    else: # logistic
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_vec, y)
    print("Huấn luyện thành công!")

    # ---- 4. Lưu lại Vectorizer và Model ----
    artifacts_dir = os.path.join(__root__, "lab_6", "artifacts")
    # artifacts_dir = os.path.join(__root__, "lab_6", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Đổi đuôi file thành .pkl
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    model_path = os.path.join(artifacts_dir, "model.pkl")

    # Sử dụng pickle.dump để lưu file
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Vectorizer đã được lưu tại: {vectorizer_path}")
    print(f"Model đã được lưu tại: {model_path}")


if __name__ == "__main__":
    run_training(vectorizer_type='count', model_type='naive_bayes')