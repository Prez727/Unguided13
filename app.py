import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, dan daftar fitur
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")

st.markdown("Isi data mahasiswa di bawah ini untuk memprediksi apakah berisiko dropout:")

# Input interaktif sesuai fitur penting
admission_grade = st.slider("Admission Grade", 0.0, 20.0, 12.0)
units_enrolled = st.slider("Mata Kuliah Semester 1 (Enrolled)", 0, 10, 5)
units_approved = st.slider("Mata Kuliah Semester 1 (Approved)", 0, 10, 5)
units_grade = st.slider("Nilai Semester 1", 0.0, 20.0, 10.0)
debtor = st.selectbox("Status Debitur", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
tuition_up_to_date = st.selectbox("Biaya Kuliah Terbayar", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
scholarship = st.selectbox("Penerima Beasiswa", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
age = st.slider("Usia Saat Masuk", 17, 50, 20)
mother_edu = st.selectbox("Pendidikan Ibu", [0, 1, 2, 3, 4, 5])
father_edu = st.selectbox("Pendidikan Ayah", [0, 1, 2, 3, 4, 5])

# Buat dataframe input dengan urutan dan nama kolom yang sinkron
input_data = pd.DataFrame([[
    admission_grade, units_enrolled, units_approved, units_grade,
    debtor, tuition_up_to_date, scholarship, age, mother_edu, father_edu
]], columns=feature_names)

# Transform dan prediksi
input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)[0]

# Output hasil prediksi
st.markdown("---")
if pred == 1:
    st.error("⚠️ Mahasiswa ini berisiko tinggi untuk dropout.")
else:
    st.success("✅ Mahasiswa ini tidak berisiko dropout.")
