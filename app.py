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
# LOAD MODEL & SCALER (Cached agar tidak reload setiap interaksi)
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("model.risiko_v1.joblib")
    scaler = joblib.load("scaler.risiko_v1.joblib")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ==========================================
# HEADER APLIKASI
# ==========================================
st.title("📊 Simulator Prediksi Risiko")
st.markdown("""
Aplikasi ini merupakan hasil operasionalisasi (deployment) model Machine Learning
yang telah dilatih pada tahap sebelumnya. Masukkan nilai fitur di bawah ini untuk
melihat hasil prediksi risiko secara real-time.
""")
st.divider()

if not model_loaded:
    st.error(
        "⚠️ File model.risiko_v1.joblib atau scaler.risiko_v1.joblib tidak ditemukan. "
        "Pastikan kedua file tersebut berada dalam satu folder dengan app.py."
    )
    st.stop()

# ==========================================
# INPUT PENGGUNA
# ==========================================
# CATATAN PENTING:
# Sesuaikan nama, jumlah, dan URUTAN input di bawah ini
# agar PERSIS SAMA dengan urutan kolom X saat model dilatih (fit).
# Jika urutan/kolom berbeda, hasil prediksi bisa salah tanpa error sama sekali.

st.subheader("Input Data Simulasi")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=17, max_value=100, value=30, step=1)
    pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, value=5000000, step=100000)
    lama_bekerja = st.number_input("Lama Bekerja (tahun)", min_value=0, max_value=50, value=3, step=1)

with col2:
    jumlah_pinjaman = st.number_input("Jumlah Pinjaman (Rp)", min_value=0, value=10000000, step=500000)
    tenor = st.slider("Tenor Cicilan (bulan)", min_value=1, max_value=60, value=12)
    riwayat_kredit = st.selectbox("Riwayat Kredit", ["Baik", "Cukup", "Buruk"])

# Encoding sederhana untuk fitur kategorikal
riwayat_map = {"Baik": 0, "Cukup": 1, "Buruk": 2}
riwayat_encoded = riwayat_map[riwayat_kredit]

st.divider()

# ==========================================
# TOMBOL PREDIKSI
# ==========================================
if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):

    # Susun input sesuai urutan fitur saat training
    input_data = np.array([[
        usia,
        pendapatan,
        lama_bekerja,
        jumlah_pinjaman,
        tenor,
        riwayat_encoded
    ]])

    # Standarisasi input menggunakan scaler yang sama dengan saat training
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediksi = model.predict(input_scaled)[0]

    # Jika model mendukung predict_proba (klasifikasi probabilistik)
    try:
        proba = model.predict_proba(input_scaled)[0]
        confidence = np.max(proba) * 100
    except AttributeError:
        proba = None
        confidence = None

    st.subheader("Hasil Simulasi")

    if prediksi == 1:
        st.error("⚠️ Status: **BERISIKO TINGGI**")
    else:
        st.success("✅ Status: **RISIKO RENDAH**")

    if confidence is not None:
        st.metric(label="Tingkat Keyakinan Model", value=f"{confidence:.2f}%")

        # Visualisasi probabilitas dengan seaborn/matplotlib
        fig, ax = plt.subplots(figsize=(5, 3))
        kelas = ["Risiko Rendah", "Risiko Tinggi"]
        sns.barplot(x=kelas, y=proba, palette=["#2ecc71", "#e74c3c"], ax=ax)
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        for i, v in enumerate(proba):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
        st.pyplot(fig)

    with st.expander("Lihat detail data input"):
        st.dataframe(
            pd.DataFrame(input_data, columns=[
                "Usia", "Pendapatan", "Lama Bekerja",
                "Jumlah Pinjaman", "Tenor", "Riwayat Kredit (encoded)"
            ]),
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Model Simulasi Risiko v1 · Dibangun dengan Streamlit · Tugas Pemodelan & Simulasi (MLOps)")