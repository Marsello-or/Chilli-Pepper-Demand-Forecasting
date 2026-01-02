import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="Chili Demand Forecast",
    page_icon="🌶️",
    layout="centered"
)

st.title("AI Chili Demand Forecaster")
st.markdown("Prediksi permintaan Cabai Rawit menggunakan **Machine Learning (XGBoost)** untuk optimasi rantai pasok.")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    try:
        model.load_model("model_cabai_xgb.json")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. INPUT USER ---
st.sidebar.header("Input Parameter")

# Input Harga
harga = st.sidebar.number_input("Rencana Harga Jual (Rp/Kg)", min_value=10000, max_value=150000, value=45000, step=1000)

# Input Hari
hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
hari_pilihan = st.sidebar.selectbox("Pilih Hari", hari_list)
# Konversi Hari ke Angka (0=Senin, 6=Minggu)
day_map = {hari: i for i, hari in enumerate(hari_list)}
day_of_week = day_map[hari_pilihan]

# Input Season (Logika One-Hot Encoding)
season_options = ["Normal", "Lebaran", "Nataru", "Idul Adha"]
season_pilihan = st.sidebar.selectbox("Musim / Event", season_options)

# --- 3. LOGIKA DATA PREPARATION ---
# Tombol Prediksi
if st.button("Prediksi Permintaan!"):
    if model:
        # Menyusun data input SAMA PERSIS seperti saat training (X_train)
        # Fitur dasar: ['Harga_Per_Kg', 'Is_Holiday_Season', 'DayOfWeek']
        
        is_holiday = 0 if season_pilihan == "Normal" else 1
        
        # Bikin Dictionary data dasar
        input_dict = {
            'Harga_Per_Kg': [harga],
            'Is_Holiday_Season': [is_holiday],
            'DayOfWeek': [day_of_week]
        }
        
        # Handle One-Hot Encoding Manual
        # Model XGBoost mengharapkan kolom: 'Nama_Season_Lebaran', 'Nama_Season_Nataru', 'Nama_Season_Idul Adha'
        # Set default 0 semua dulu
        for s in ["Lebaran", "Nataru", "Idul Adha"]:
            col_name = f"Nama_Season_{s}"
            input_dict[col_name] = [1 if season_pilihan == s else 0]

        # Convert ke DataFrame
        input_df = pd.DataFrame(input_dict)
        
        try:
            # Prediksi
            prediction = model.predict(input_df)[0]
            
            # --- TAMPILAN HASIL ---
            st.success(f"Prediksi Permintaan: **{int(prediction)} Kg**")
            
            # Insight Bisnis
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Rekomendasi Stok:**\nSiapkan **{int(prediction * 1.1)} Kg** (Buffer 10%)")
            with col2:
                revenue_est = int(prediction * harga)
                st.metric("Estimasi Omzet", f"Rp {revenue_est:,}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan format data: {e}")
            st.warning("Tips: Pastikan fitur input di app.py sama persis dengan X_train di Colab.")

    else:
        st.error("Model belum dimuat. Pastikan file .json ada.")