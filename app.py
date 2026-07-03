import streamlit as st
import numpy as np
import joblib

# 1. Load Model dan Scaler di bagian atas (Prinsip Inference)
model = joblib.load('model_risiko_v1.joblib')
scaler = joblib.load('scaler_risiko_v1.joblib')

st.title("Simulator Risiko Mesin")
st.write("Masukkan data sensor untuk memprediksi risiko kegagalan sistem.")

# 2. Input data dari pengguna
suhu_input = st.number_input("Masukkan Suhu Mesin:", min_value=0)
getaran_input = st.number_input("Masukkan Getaran Mesin:", min_value=0)

# 3. Monitoring Drift (Kesehatan Model) sesuai panduan modul
if suhu_input > 120 or suhu_input < 10:
    st.warning("Input di luar jangkauan data latihan. Hasil simulasi mungkin tidak akurat!")

# 4. Tombol Prediksi
if st.button("Prediksi Risiko"):
    # Format data sesuai kebutuhan model [Suhu, Getaran]
    data_baru = np.array([[suhu_input, getaran_input]])
    
    # Transformasi (Normalisasi) menggunakan scaler yang sudah di-load
    data_baru_scaled = scaler.transform(data_baru)
    
    # Lakukan prediksi
    hasil_simulasi = model.predict(data_baru_scaled)
    
    # Tampilkan hasil
    st.success(f"Skor Risiko Kegagalan: {hasil_simulasi[0]:.2f}")