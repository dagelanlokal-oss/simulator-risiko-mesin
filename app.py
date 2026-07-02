## Langkah 3: Implementasi UI Interaktif (Konsep Streamlit)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================
# DATA CONTOH
# ==========================

X = np.array([
    [10,10],
    [20,10],
    [30,20],
    [40,30],
    [50,40]
])

y = np.array([100,130,170,220,280])

# ==========================
# LATIH MODEL
# ==========================

model = LinearRegression()
model.fit(X, y)

# ==========================
# BASELINE
# ==========================

baseline_input = pd.DataFrame([[10,10]], columns=[
    "Iklan",
    "Diskon"
])

baseline_pred = model.predict(baseline_input)[0]

# ==========================
# FUNGSI SIMULASI
# ==========================

def run_simulation(iklan, diskon):

    input_data = pd.DataFrame([[iklan,diskon]], columns=[
        "Iklan",
        "Diskon"
    ])

    hasil_pred = model.predict(input_data)[0]

    delta = hasil_pred - baseline_pred

    return hasil_pred, delta

import streamlit as st

st.title("🚀🚀 Simulator Kebijakan Ekonomi")
st.header("Analisis Skenario What-If")
st.write("Aplikasi ini mensimulasikan dampak perubahan variabel terhadap keuntungan.")

# Membuat bilah samping
st.sidebar.header("Variabel Kontrol (Intervensi)")
# Widget Slider: (Label, Nilai Min, Nilai Max, Nilai Default)
anggaran_iklan = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 100, 10)
persen_diskon = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 5)

col1, col2 = st.columns(2)
# Menampilkan angka besar dengan indikator kenaikan (Delta)
col1.metric(label="Prediksi Keuntungan", value="Rp 150 Juta", delta="12 Juta")
col2.metric(label="Risiko Stok", value="5%", delta="-2%", delta_color="inverse")

import joblib # Jika model disimpan sebagai file .pkl

# 1. Load Model (Contoh model regresi minggu ke-4)
model = joblib.load('model_risiko_v1.joblib')

# 2. UI Interaksi
st.sidebar.title("Input Kebijakan")
iklan = st.sidebar.slider("Iklan", 0, 50, 10)
diskon = st.sidebar.slider("Diskon", 0, 20, 5)

# 3. Prediksi (Logika What-If)
# Mengubah input slider menjadi format matriks X [Samples, Features]
input_data = np.array([[iklan, diskon]])
prediksi = model.predict(input_data)[0]

# 4. Tampilkan
st.subheader("Hasil Simulasi")
st.success(f"Keuntungan yang diprediksi: Rp {prediksi:.2f} Juta")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# Logika grafik perbandingan baseline vs intervensi
ax.bar(['Baseline', 'Skenario Baru'], [100, prediksi])
st.pyplot(fig)