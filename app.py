import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Simulator Prediksi Risiko",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("model_risiko_v1.joblib")
    scaler = joblib.load("scaler_risiko_v1.joblib")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ==========================================
# HEADER
# ==========================================
st.title("📊 Simulator Prediksi Risiko")
st.markdown("Aplikasi simulasi prediksi risiko mesin menggunakan Machine Learning.")
st.divider()

if not model_loaded:
    st.error("⚠️ File model atau scaler tidak ditemukan.")
    st.stop()

# ==========================================
# INPUT PENGGUNA (Hanya 2 fitur sesuai scaler)
# ==========================================
st.subheader("Input Data Simulasi")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=17, max_value=100, value=30, step=1)

with col2:
    pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, value=5000000, step=100000)

st.divider()

# ==========================================
# PREDIKSI
# ==========================================
if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):

    # Input sesuai dengan 2 fitur yang diharapkan scaler
    input_data = pd.DataFrame({
        "Usia": [usia],
        "Pendapatan": [pendapatan]
    })

    st.write(f"Jumlah fitur input: {input_data.shape[1]}")

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediksi = model.predict(input_scaled)[0]

    try:
        proba = model.predict_proba(input_scaled)[0]
        confidence = np.max(proba) * 100
    except:
        confidence = None

    st.subheader("Hasil Simulasi")

    if prediksi == 1:
        st.error("⚠️ Status: **BERISIKO TINGGI**")
    else:
        st.success("✅ Status: **RISIKO RENDAH**")

    if confidence is not None:
        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

    with st.expander("Detail Input"):
        st.dataframe(input_data, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Simulator Risiko Mesin v1 · Streamlit + Scikit-learn")