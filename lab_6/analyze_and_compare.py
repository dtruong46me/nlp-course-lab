import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc

# --- 1. CÁC HÀM PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU (EDA) ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm tiền xử lý dữ liệu, giống các file trước."""
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['text'] = df['text'].astype(str).str.replace(r'^Subject: ', '', regex=True, n=1)
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def plot_top_words(df: pd.DataFrame):
    """Vẽ biểu đồ các từ phổ biến nhất trong email Spam và Ham."""
    print("\n--- Phân tích các từ phổ biến nhất ---")
    
    spam_corpus = df[df['label'] == 'spam']['text']
    ham_corpus = df[df['label'] == 'ham']['text']
    
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    
    # Phân tích từ trong email SPAM
    spam_words = vectorizer.fit_transform(spam_corpus)
    spam_counts = pd.DataFrame(spam_words.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False) # type: ignore
    
    # Print top words
    print("20 từ phổ biến nhất trong email SPAM:")
    print(spam_counts)

    # Phân tích từ trong email HAM
    ham_words = vectorizer.fit_transform(ham_corpus)
    ham_counts = pd.DataFrame(ham_words.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False) # type: ignore
    
    print("\n20 từ phổ biến nhất trong email HAM:")
    print(ham_counts)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.barplot(x=spam_counts.values, y=spam_counts.index, ax=axes[0], palette='Reds_d')
    axes[0].set_title('20 từ phổ biến nhất trong email SPAM', fontsize=16)
    axes[0].set_xlabel('Tần suất')
    
    sns.barplot(x=ham_counts.values, y=ham_counts.index, ax=axes[1], palette='Blues_d')
    axes[1].set_title('20 từ phổ biến nhất trong email HAM', fontsize=16)
    axes[1].set_xlabel('Tần suất')
    
    plt.tight_layout()
    plt.show()

def generate_word_clouds(df: pd.DataFrame):
    """Tạo và hiển thị Word Cloud cho email Spam và Ham."""
    spam_text = " ".join(email for email in df[df['label'] == 'spam']['text'])
    ham_text = " ".join(email for email in df[df['label'] == 'ham']['text'])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    spam_wordcloud = WordCloud(stopwords=None, background_color='white', width=800, height=400).generate(spam_text)
    axes[0].imshow(spam_wordcloud, interpolation='bilinear')
    axes[0].set_title('Word Cloud - SPAM', fontsize=20)
    axes[0].axis('off')

    # Print spam words
    print("\nMột số từ xuất hiện trong email SPAM:")
    print(", ".join(list(spam_wordcloud.words_.keys())[:20]))

    ham_wordcloud = WordCloud(stopwords=None, background_color='white', width=800, height=400).generate(ham_text)
    axes[1].imshow(ham_wordcloud, interpolation='bilinear')
    axes[1].set_title('Word Cloud - HAM', fontsize=20)
    axes[1].axis('off')

    # Print ham words
    print("\nMột số từ xuất hiện trong email HAM:")
    print(", ".join(list(ham_wordcloud.words_.keys())[:20]))
    
    
    plt.show()


# --- 2. HÀM CHÍNH ĐỂ SO SÁNH CÁC MÔ HÌNH ---

def run_model_comparison():
    """
    Hàm chính để tải dữ liệu, huấn luyện, đánh giá, và so sánh các mô hình.
    """
    # Tải và chuẩn bị dữ liệu
    print("--- Tải và chuẩn bị dữ liệu cho việc huấn luyện ---")
    __root__ = os.getcwd()
    data_path = os.path.join(__root__, "lab_6", "spam_data", "spam_ham_dataset.csv")
    data = preprocess_data(pd.read_csv(data_path))
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label_num'], test_size=0.2, random_state=42, stratify=data['label_num']
    )
    
    # Vector hóa dữ liệu
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Định nghĩa các mô hình cần so sánh
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True), # probability=True để vẽ ROC
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = {}
    
    plt.figure(figsize=(10, 8)) # Chuẩn bị figure cho ROC
    
    # Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        print(f"\n--- Đang huấn luyện: {name} ---")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        # Lưu kết quả
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        # Tính toán ROC
        y_proba = model.predict_proba(X_test_vec)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
    print("\n--- KẾT QUẢ SO SÁNH ---")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # --- Trực quan hóa kết quả so sánh ---
    # Biểu đồ ROC
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('So sánh đường cong ROC của các mô hình', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Biểu đồ cột F1-Score và Accuracy
    results_df.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title('So sánh Accuracy và F1-Score', fontsize=16)
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--')
    plt.show()


if __name__ == "__main__":
    # --- Chạy phân tích đặc trưng dữ liệu (EDA) ---
    __root__ = os.getcwd()
    data_path = os.path.join(__root__, "lab_6", "spam_data", "spam_ham_dataset.csv")
    df = preprocess_data(pd.read_csv(data_path))
    plot_top_words(df)
    generate_word_clouds(df)
    
    # --- Chạy so sánh mô hình ---
    run_model_comparison()