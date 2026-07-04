import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. PERSIAPAN MODEL & FUNGSI SPK (MCDM)
# ==========================================
# Ingat: Path diubah karena kita menaruh model di folder 'models/'
model = joblib.load('models/model_risiko_v1.joblib')
scaler = joblib.load('models/scaler_risiko_v1.joblib')

# Fungsi SAW (Simple Additive Weighting) Sederhana
def jalankan_saw(matriks, bobot):
    # Asumsi Kriteria: 
    # C1: Risiko (Cost - Semakin kecil semakin baik)
    # C2: Biaya Pemeliharaan (Cost)
    # C3: Efisiensi (Benefit - Semakin besar semakin baik)
    norm_matriks = np.zeros_like(matriks, dtype=float)
    
    # Normalisasi Cost (Min / X)
    norm_matriks[:, 0] = np.min(matriks[:, 0]) / matriks[:, 0]
    norm_matriks[:, 1] = np.min(matriks[:, 1]) / matriks[:, 1]
    # Normalisasi Benefit (X / Max)
    norm_matriks[:, 2] = matriks[:, 2] / np.max(matriks[:, 2])
    
    # Perhitungan Skor Akhir (Perkalian dengan bobot)
    skor_akhir = np.dot(norm_matriks, bobot)
    return skor_akhir

# ==========================================
# 2. ANTARMUKA STREAMLIT (UI)
# ==========================================
st.title("Simulator Risiko Mesin (UAS Version)")
st.write("Sistem Cerdas Integrasi ML (Prediksi) dan SPK (Rekomendasi Tindakan)")

# Menggunakan Slider untuk Interaksi (Live Intervention)
st.subheader("1. Intervensi Skenario (What-If)")
suhu_input = st.slider("Atur Suhu Mesin:", min_value=0, max_value=150, value=85)
getaran_input = st.slider("Atur Getaran Mesin:", min_value=0, max_value=20, value=7)

# Monitoring Drift
if suhu_input > 120 or suhu_input < 10:
    st.warning("WARNING: Input di luar jangkauan data latihan. Hasil simulasi mungkin tidak akurat!")

# ==========================================
# 3. ALIRAN DATA TERINTEGRASI (INFERENCE -> MCDM)
# ==========================================
if st.button("Jalankan Simulasi Sistem"):
    # --- PROSES MACHINE LEARNING ---
    data_baru = np.array([[suhu_input, getaran_input]])
    data_baru_scaled = scaler.transform(data_baru)
    
    prediksi_risiko = model.predict(data_baru_scaled)[0]
    st.info(f"**Prediksi Skor Risiko (ML): {prediksi_risiko:.2f}**")

    # --- PROSES SPK (MCDM) ---
    st.subheader("2. Rekomendasi Tindakan (SPK SAW)")
    # Membuat matriks keputusan [Risiko (ML), Biaya, Efisiensi]
    # Alternatif 1: Mesin A (Mesin yang sedang disimulasikan)
    # Alternatif 2: Mesin B (Pembanding statis)
    # Alternatif 3: Mesin C (Pembanding statis)
    matriks_x = np.array([
        [prediksi_risiko, 500, 80],  # Mesin A: Risiko dari ML, biaya 500, efisiensi 80
        [60.0, 450, 75],             # Mesin B: Statis
        [85.0, 300, 60]              # Mesin C: Statis
    ])
    
    # Bobot Keputusan (Asumsi dari AHP/Pakar): Risiko(50%), Biaya(30%), Efisiensi(20%)
    bobot_ahp = np.array([0.5, 0.3, 0.2])
    
    # Hitung Ranking
    skor_akhir = jalankan_saw(matriks_x, bobot_ahp)
    
    # Menentukan Rekomendasi
    alternatif = ["Mesin A (Saat Ini)", "Mesin B", "Mesin C"]
    ranking = sorted(zip(alternatif, skor_akhir), key=lambda x: x[1], reverse=True)
    
    st.write(f"Berdasarkan hitungan SPK, prioritas perbaikan tertinggi jatuh pada: **{ranking[0][0]}** dengan skor {ranking[0][1]:.3f}")

    # ==========================================
# 4. MODUL TRANSPARANSI (XAI - SHAP)
# ==========================================
    st.subheader("3. Mengapa Hasilnya Demikian?")
    st.write("Kontribusi fitur Suhu dan Getaran terhadap Prediksi Risiko:")
    
    # Menyiapkan Explainer SHAP
    # (Catatan: Menggunakan explainer dasar untuk linear regression)
    explainer = shap.LinearExplainer(model, scaler.transform(np.array([[60,2],[70,4],[80,6],[90,8],[100,10]])))
    shap_values = explainer(data_baru_scaled)
    
    # Menggambar plot waterfall
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.clf()