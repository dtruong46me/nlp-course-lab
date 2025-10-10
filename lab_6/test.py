# ...existing code...
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Hàm tiền xử lý phải giống hệt trong file train.py
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Thực hiện các bước tiền xử lý dữ liệu email."""
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['text'] = df['text'].astype(str) # type: ignore
    df['text'] = df['text'].str.replace(r'[\n\r\t]', ' ', regex=True) # type: ignore
    df['text'] = df['text'].str.replace(r'^Subject: ', '', regex=True, n=1) # type: ignore
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def run_evaluation():
    """
    Hàm chính để thực hiện đánh giá và so sánh các mô hình.
    """
    # ---- 1. Tải và chuẩn bị dữ liệu ----
    print("Tải và chuẩn bị dữ liệu...")
    __root__ = os.getcwd()
    data_path = os.path.join(__root__, "lab_6", "spam_data", "spam_ham_dataset.csv")
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{data_path}'")
        return

    # create output folder
    output_dir = os.path.join(__root__, "lab_6", "output")
    os.makedirs(output_dir, exist_ok=True)

    def _sanitize(name: str) -> str:
        # simple sanitizer for filenames
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

    data = pd.read_csv(data_path)
    processed_data = preprocess_data(data)

    X = processed_data['text']
    y = processed_data['label_num']
    
    # Chia dữ liệu thành tập train và test để đánh giá khách quan
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Sử dụng CountVectorizer (hoặc vectorizer bạn đã dùng trong train.py)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ---- 2. Định nghĩa các mô hình cần so sánh ----
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    results = {}
    
    # Prepare ROC figure
    roc_fig, roc_ax = plt.subplots(figsize=(10, 8))
    
    # ---- 3. Huấn luyện và đánh giá từng mô hình ----
    for name, model in models.items():
        print(f"\n--- Đang huấn luyện và đánh giá: {name} ---")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        # some models may not support predict_proba; handle gracefully
        try:
            y_proba = model.predict_proba(X_test_vec)[:, 1]
        except Exception:
            # fallback: use decision_function if available, else use predictions
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test_vec)
                # normalize to [0,1]
                min_s, max_s = scores.min(), scores.max()
                if max_s > min_s:
                    y_proba = (scores - min_s) / (max_s - min_s)
                else:
                    y_proba = (scores - min_s)
            else:
                y_proba = y_pred

        # Lưu kết quả
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "f1_score": f1}
        
        # In báo cáo chi tiết
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Vẽ ma trận nhầm lẫn
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, 
                                              display_labels=['Ham', 'Spam'], 
                                              cmap='Blues', ax=ax)
        ax.set_title(f'Ma trận nhầm lẫn - {name}')
        # save confusion matrix
        fname = f"confusion_{_sanitize(name)}.png"
        fig_path = os.path.join(output_dir, fname)
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved confusion matrix to: {fig_path}")
        plt.close(fig)

        # Tính toán và vẽ đường cong ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # ---- 4. Visualize so sánh ----
    # Hoàn thiện biểu đồ ROC
    roc_ax.plot([0, 1], [0, 1], 'k--', label='Chance')
    roc_ax.set_xlim([0.0, 1.0]) # type: ignore
    roc_ax.set_ylim([0.0, 1.05]) # type: ignore
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('So sánh đường cong ROC các mô hình')
    roc_ax.legend(loc="lower right")
    roc_ax.grid(True)
    roc_path = os.path.join(output_dir, "roc_comparison.png")
    roc_fig.savefig(roc_path, bbox_inches='tight')
    print(f"Saved ROC comparison to: {roc_path}")
    plt.close(roc_fig)

    # Tạo dataframe từ kết quả để vẽ biểu đồ cột
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Vẽ biểu đồ cột so sánh
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y=['accuracy', 'f1_score'], kind='bar', ax=ax2,
                    title='So sánh hiệu năng giữa các mô hình')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticklabels(results_df['Model'], rotation=0)
    ax2.grid(axis='y', linestyle='--')
    comp_path = os.path.join(output_dir, "model_comparison.png")
    fig2.savefig(comp_path, bbox_inches='tight')
    print(f"Saved model comparison chart to: {comp_path}")
    plt.close(fig2)


if __name__ == "__main__":
    run_evaluation()
# ...existing code...