import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, feature names
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 Prediksi Dropout Mahasiswa")

# Input interaktif
admission_grade = st.slider("Admission Grade", 0.0, 20.0, 12.0)
units_enrolled = st.slider("Mata Kuliah Semester 1 (Enrolled)", 0, 10, 5)
units_approved = st.slider("Mata Kuliah Semester 1 (Approved)", 0, 10, 5)
units_grade = st.slider("Nilai Semester 1", 0.0, 20.0, 10.0)
debtor = st.selectbox("Status Debitur", [0, 1])
tuition_up_to_date = st.selectbox("Biaya Kuliah Terbayar", [0, 1])
scholarship = st.selectbox("Penerima Beasiswa", [0, 1])
age = st.slider("Usia Saat Masuk", 17, 50, 20)
mother_edu = st.selectbox("Pendidikan Ibu", [0, 1, 2, 3, 4, 5])
father_edu = st.selectbox("Pendidikan Ayah", [0, 1, 2, 3, 4, 5])

# Buat dataframe input
input_data = pd.DataFrame([[
    admission_grade, units_enrolled, units_approved, units_grade,
    debtor, tuition_up_to_date, scholarship, age, mother_edu, father_edu
]], columns=[
    "Admission grade", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Debtor", "Tuition fees up to date", "Scholarship holder",
    "Age at enrollment", "Mother's qualification", "Father's qualification"
])

# Scaling dan prediksi
input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)[0]

# Output
if pred == 1:
    st.error("⚠️ Mahasiswa berisiko Dropout")
else:
    st.success("✅ Mahasiswa tidak berisiko Dropout")
