import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load model, scaler, dan data evaluasi
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# Hitung akurasi
X_test_scaled = scaler.transform(X_test)
model_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

# Ambil fitur penting
feature_importance = dict(zip(X_test.columns, model.feature_importances_))
sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

# Konfigurasi halaman
st.set_page_config(page_title="Sistem Peringatan Dropout Mahasiswa", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 Sistem Peringatan Dini Dropout Mahasiswa</h1>", unsafe_allow_html=True)

# Layout 3 kolom
col1, col2, col3 = st.columns([1.2, 1.2, 1])

# 🔹 Input Mahasiswa
with col1:
    st.subheader("📘 Input Data Mahasiswa")
    st.markdown("### Akademik")
    units_approved_2 = st.slider("SKS Lulus (Sem 2)", 0, 10, 1)
    grade_2 = st.slider("Nilai Rata-rata (Sem 2)", 0.0, 20.0, 1.32)
    units_approved_1 = st.slider("SKS Lulus (Sem 1)", 0, 10, 2)
    grade_1 = st.slider("Nilai Rata-rata (Sem 1)", 0.0, 20.0, 4.99)

    st.markdown("### Finansial")
    tuition = st.selectbox("Uang Kuliah Lancar?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    scholarship = st.selectbox("Penerima Beasiswa?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    debtor = st.selectbox("Memiliki Utang?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    age = st.slider("Usia Saat Masuk", 17, 50, 20)

# 🔍 Hasil Prediksi
with col2:
    st.subheader("📊 Hasil Prediksi")

    input_data = pd.DataFrame([[
        grade_1,
        units_approved_1,
        grade_2,
        units_approved_2,
        age,
        debtor,
        tuition,
        scholarship
    ]], columns=X_test.columns)

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]  # Probabilitas dropout
    pred = model.predict(input_scaled)[0]

    st.markdown(f"<h2 style='text-align:center;'>Tingkat Risiko Dropout: <span style='color:red;'>{prob*100:.2f}%</span></h2>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown("<div style='background-color:#ffcccc;padding:20px;border-radius:10px'><h3 style='color:red;'>⚠️ Mahasiswa Berisiko DROPOUT</h3><p>💡 Segera lakukan konseling akademik dan evaluasi dukungan finansial.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color:#ccffcc;padding:20px;border-radius:10px'><h3 style='color:green;'>✅ Mahasiswa tidak berisiko dropout</h3></div>", unsafe_allow_html=True)

# 📈 Performa Model
with col3:
    st.subheader("📈 Performa Model")
    st.metric("Akurasi Model", f"{model_accuracy * 100:.2f}%")

    st.markdown("### Fitur Terpenting")
    for feat, score in sorted_features.items():
        st.markdown(f"- **{feat}**: `{score:.4f}`")
