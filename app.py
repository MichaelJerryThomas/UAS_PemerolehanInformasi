import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict, Counter
import nltk

# --- Setup ---
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# --- Preprocessing ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

# --- Indexing ---
inverted_index = defaultdict(list)
document_tokens = {}

st.info("Membangun indeks dan mempersiapkan data, harap tunggu...")

fileids = reuters.fileids()
for fileid in fileids:
    import nltk
    nltk.download('reuters')
    nltk.download('punkt')
    nltk.download('stopwords')
    content = reuters.raw(fileid)
    tokens = preprocess_text(content)
    document_tokens[fileid] = tokens
    for term in set(tokens):
        inverted_index[term].append(fileid)

# --- Klasifikasi ---
document_labels_nltk = []
document_data_nltk = []

for fileid in fileids:
    document_data_nltk.append(reuters.raw(fileid))
    document_labels_nltk.append(reuters.categories(fileid))

label_counts = Counter([label for sublist in document_labels_nltk for label in sublist])
top_10_labels = [label for label, count in label_counts.most_common(10)]

filtered_data = []
filtered_labels = []
for i, doc_labels in enumerate(document_labels_nltk):
    relevant_labels = [label for label in doc_labels if label in top_10_labels]
    if relevant_labels:
        filtered_data.append(document_data_nltk[i])
        filtered_labels.append(relevant_labels[0])

X_train, X_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

# --- Search Engine ---
def search(query):
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    matching_docs = set(inverted_index.get(query_tokens[0], []))
    for term in query_tokens[1:]:
        if term in inverted_index:
            matching_docs.intersection_update(inverted_index[term])
        else:
            return []
    return list(matching_docs)

def get_relevant_docs_by_category(category):
    return reuters.fileids(category)

def calculate_metrics(retrieved, relevant):
    tp = len(set(retrieved) & set(relevant))
    fp = len(set(retrieved) - set(relevant))
    fn = len(set(relevant) - set(retrieved))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy, precision, recall

def evaluate_classification_query(query):
    tp = sum((y == query and p == query) for y, p in zip(y_test, y_pred_svm))
    fp = sum((y != query and p == query) for y, p in zip(y_test, y_pred_svm))
    fn = sum((y == query and p != query) for y, p in zip(y_test, y_pred_svm))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy, precision, recall

# --- Streamlit Interface ---
st.title("Search Engine & Klasifikasi Dokumen (Reuters Dataset)")

queries = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat']
selected_query = st.selectbox("Pilih Query/Kategori:", queries)

if st.button("Evaluasi Query"):
    st.subheader("🔍 Evaluasi Search Engine")
    retrieved_docs = search(selected_query)
    relevant_docs = get_relevant_docs_by_category(selected_query)
    se_acc, se_prec, se_rec = calculate_metrics(retrieved_docs, relevant_docs)
    st.write(f"Akurasi: {se_acc:.4f}")
    st.write(f"Presisi: {se_prec:.4f}")
    st.write(f"Recall: {se_rec:.4f}")

    st.subheader("📊 Evaluasi Klasifikasi (SVM)")
    cls_acc, cls_prec, cls_rec = evaluate_classification_query(selected_query)
    st.write(f"Akurasi: {cls_acc:.4f}")
    st.write(f"Presisi: {cls_prec:.4f}")
    st.write(f"Recall: {cls_rec:.4f}")
