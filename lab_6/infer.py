# infer.py
import os
import pickle
from typing import Tuple, Dict, Any

# Các hằng số trỏ đến file đã lưu
__root__ = os.getcwd()
ARTIFACTS_DIR = os.path.join(__root__, "lab_6", "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")

def load_artifacts() -> Tuple[Any, Any]:
    """Tải vectorizer và model đã được huấn luyện từ đĩa."""
    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        print("Tải model và vectorizer từ file pickle thành công.")
        return vectorizer, model
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file model/vectorizer.")
        print("Vui lòng chạy file train.py trước để huấn luyện và lưu model.")
        return None, None

def classify_email(email_text: str, vectorizer: Any, model: Any) -> Dict[str, Any]:
    """
    Phân loại một email là spam hay ham.
    """
    # Xóa "Subject: " nếu có để đồng bộ với dữ liệu train
    processed_text = email_text.replace('Subject: ', '', 1)
    
    # Chuyển đổi email text thành vector
    email_vec = vectorizer.transform([processed_text])
    
    # Dự đoán
    prediction = model.predict(email_vec)[0]
    prediction_proba = model.predict_proba(email_vec)[0]
    
    label_map = {0: "HAM", 1: "SPAM"}
    
    return {
        "prediction_label": label_map[prediction],
        "spam_confidence": prediction_proba[1] 
    }

if __name__ == "__main__":
    vectorizer, model = load_artifacts()
    
    if vectorizer and model:
        # ---- Dữ liệu mẫu để kiểm thử ----
        sample_emails = [
            "Congratulations! You've won a $1,000 Walmart gift card. Go to http://example.com to claim now.",
            "Hi team, let's meet tomorrow at 10 AM to discuss the project update. Please be prepared.",
        ]
        
        print("\n--- Bắt đầu phân loại email mẫu ---")
        for i, email in enumerate(sample_emails):
            result = classify_email(email, vectorizer, model)
            print(f"\nEmail {i+1}: \"{email[:60]}...\"")
            print(f"  -> ✅ Dự đoán: {result['prediction_label']}")
            print(f"  -> 🎯 Độ tin cậy là SPAM: {result['spam_confidence']:.2%}")
        print("\n-------------------------------------")