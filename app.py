import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("🎓 Prediksi Dropout Mahasiswa")

user_input = {}
for feature in feature_names:
    if "grade" in feature or "rate" in feature or "GDP" in feature:
        user_input[feature] = st.number_input(feature, value=0.0)
    else:
        user_input[feature] = st.number_input(feature, value=0)

input_df = pd.DataFrame([user_input])

input_df = input_df[feature_names]

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

if prediction == 1:
    st.error("⚠️ Mahasiswa berisiko Dropout")
else:
    st.success("✅ Mahasiswa tidak berisiko Dropout")
