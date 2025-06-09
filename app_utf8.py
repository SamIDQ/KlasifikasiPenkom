import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from gensim.models import Word2Vec

# Konfigurasi tampilan halaman
st.set_page_config(
    page_title="Sentiment Analyzer Gojek",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi label sentimen
def label_sentiment(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

# Highlight kata kunci penting berdasarkan TF-IDF score
def get_keywords(text, tfidf_vectorizer, top_n=5):
    tfidf_matrix = tfidf_vectorizer.transform([text])
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords

st.markdown("""
    <style>
    .main { background-color: #F9FBFC; }
    .st-bb { background-color: #FFFFFF !important; }
    .st-cf { font-size: 16px; }
    .stButton>button { background-color: #0073e6; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4F1 Analisis Sentimen Review Aplikasi Gojek")

# # Load dataset
# file_path = Path(r"db\GojekReviewer_final.csv")
# if not file_path.exists():
#     st.error(f"\u274c File tidak ditemukan di path: {file_path}")
#     st.stop()

# df = pd.read_csv(file_path)
# df['sentimen'] = df['score'].apply(label_sentiment)

# Load dataset
file_path = Path("db") / "GojekReviewer_final.csv"
if not file_path.exists():
    st.error(f"❌ File tidak ditemukan di path: {file_path.resolve()}")
    st.stop()

df = pd.read_csv(file_path)
df['sentimen'] = df['score'].apply(label_sentiment)

if 'cleaned' not in df.columns or df['cleaned'].isnull().any():
    st.warning("Kolom 'cleaned' tidak ditemukan atau ada nilai kosong. Pastikan kolom 'cleaned' tersedia.")
    st.stop()

# Sidebar - Model dan input pengguna
with st.sidebar:
    st.header("🔧 Konfigurasi")
    model_choice = st.selectbox("Pilih model analisis sentimen:", ["SVM", "LSTM"])
    input_text = st.text_area("Masukkan ulasan Anda:")
    predict_btn = st.button("Prediksi")

# Sampling data
df_sample = df.sample(n=2000, random_state=42)
texts = df_sample['cleaned']
labels = df_sample['sentimen']
le = LabelEncoder()
y = le.fit_transform(labels)

if model_choice == "SVM":
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred_probs = svm.predict_proba(X_test)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Evaluasi Model SVM")
        report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🌐 Cross Validation")
        cv_scores = cross_val_score(svm, X, y, cv=5)
        st.metric(label="Rata-rata Akurasi CV (5-fold)", value=f"{cv_scores.mean():.2%}")
        st.line_chart(pd.Series(cv_scores, name="Akurasi Fold"))

    with col2:
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)

        st.subheader("🔢 Top 10 Kata TF-IDF")
        tfidf_features = pd.DataFrame(X_train.toarray(), columns=tfidf.get_feature_names_out())
        top_words = tfidf_features.mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_words)

elif model_choice == "LSTM":
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=100)
    y_cat = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, stratify=y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=64),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Evaluasi Model LSTM")
        report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)

    tokenized = [text.split() for text in texts]
    w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=3, min_count=1, workers=1, epochs=50)

    st.subheader("🔢 Representasi Word2Vec")
    st.write("Contoh vektor kata 'gojek':")
    if 'gojek' in w2v_model.wv.key_to_index:
        gojek_vec = w2v_model.wv['gojek']
        st.code(gojek_vec[:10])
    else:
        st.warning("Kata 'gojek' tidak ditemukan dalam model Word2Vec.")

# Prediksi ulasan baru
if predict_btn and input_text.strip():
    st.sidebar.success("Prediksi berhasil!")

    if model_choice == "SVM":
        input_vec = tfidf.transform([input_text])
        pred_label_num = svm.predict(input_vec)[0]
        pred_proba = svm.predict_proba(input_vec)[0]
        pred_label = le.inverse_transform([pred_label_num])[0]
        confidence = pred_proba[pred_label_num]

    elif model_choice == "LSTM":
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=100)
        probs = model.predict(padded)[0]
        pred_label_num = np.argmax(probs)
        pred_label = le.inverse_transform([pred_label_num])[0]
        confidence = probs[pred_label_num]

    st.markdown("---")
    st.header("🔮 Hasil Prediksi Ulasan Baru")
    st.success(f"Sentimen: {pred_label}")
    st.info(f"Confidence: {confidence:.2%}")

    if model_choice == "SVM":
        keywords = get_keywords(input_text, tfidf, top_n=5)
        st.write("**Kata Kunci Penting:**")
        st.write(", ".join(keywords))
    else:
        st.write("_Highlight kata kunci tidak tersedia untuk model LSTM._")
