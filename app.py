import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open("logreg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Prediksi Dropout Mahasiswa 🎓")

ipk = st.number_input("IPK", 0.0, 4.0, 3.0)
sks = st.number_input("Jumlah SKS", 0, 150, 100)
kehadiran = st.slider("Persentase Kehadiran (%)", 0, 100, 80)
pendapatan = st.number_input("Pendapatan Orang Tua (juta)", 0, 100, 10)

input_data = pd.DataFrame([[ipk, sks, kehadiran, pendapatan]],
                          columns=["ipk","sks","kehadiran","pendapatan"])

input_scaled = scaler.transform(input_data)


pred = model.predict(input_scaled)[0]

if pred == 1:
    st.error("⚠️ Mahasiswa berisiko Dropout")
else:
    st.success("✅ Mahasiswa tidak berisiko Dropout")
