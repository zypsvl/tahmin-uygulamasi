# app.py (DOĞRU VE SADE KOD)

import streamlit as st
import pandas as pd
import joblib
from src.data_processing import normalize_columns, process_for_modeling

# --- Sayfa Ayarları ---
st.set_page_config(layout="wide", page_title="Dinamik Tahmin Modeli")
st.title("🤖 Dinamik Model Test Uygulaması")
st.info("Bu uygulama, dışarıdan yüklediğiniz model ve veri dosyaları ile tahmin yapmanızı sağlar.")

# --- Yardımcı Fonksiyonlar ---
def read_data_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8-sig', header=0)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Desteklenmeyen dosya formatı. Lütfen .csv veya .xlsx dosyası yükleyin.")
            return None
        return normalize_columns(df)
    except Exception as e:
        st.error(f"Veri dosyası okunurken bir hata oluştu: {e}")
        return None

def align_columns(df, model_columns):
    missing_cols = set(model_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    return df[model_columns]

# --- Ana Arayüz ---
st.header("Adım 1: Model ve Sütun Dosyalarını Yükleyin")

col1, col2 = st.columns(2)
with col1:
    up_model = st.file_uploader("Eğitilmiş Model (.pkl)", type="pkl", key="model_upload")
with col2:
    up_cols = st.file_uploader("Modele Ait Sütun Listesi (.pkl)", type="pkl", key="cols_upload")

st.header("Adım 2: Tahmin Yapılacak Veri Dosyasını Yükleyin")
up_data = st.file_uploader(
    "Test verisi (CSV veya Excel):",
    type=["csv", "xlsx", "xls"],
    key="data_upload"
)

st.markdown("---")

if st.button("🚀 Tahminleri Başlat", type="primary", use_container_width=True):
    if up_model and up_cols and up_data:
        try:
            model = joblib.load(up_model)
            model_cols = joblib.load(up_cols)

            with st.spinner("Veri dosyası okunuyor ve işleniyor..."):
                df_raw = read_data_file(up_data)
            
            if df_raw is not None:
                with st.spinner("Tahminler yapılıyor..."):
                    X_ready = process_for_modeling(df_raw, model_cols, create_targets=False, fill_na_method='fill_zero')
                    
                    if X_ready.empty:
                        st.error("Ön işleme sonrası tahmin edilecek veri kalmadı. Lütfen dosyanızı kontrol edin.")
                    else:
                        X_aligned = align_columns(X_ready, model_cols)
                        
                        y_pred = model.predict(X_aligned)
                        y_proba = model.predict_proba(X_aligned)
                        
                        df_res = pd.DataFrame({
                            "Tahmin (0: OK, 1: Hatalı)": y_pred,
                            "OK Olma Olasılığı": y_proba[:, 0],
                            "Hatalı Olma Olasılığı": y_proba[:, 1]
                        }, index=X_aligned.index)
                        
                        st.success(f"Tahminler {len(df_res)} satır için başarıyla oluşturuldu!")
                        st.dataframe(df_res)

                        st.subheader("Genel Ortalama Olasılıklar")
                        avg_ok_proba = df_res["OK Olma Olasılığı"].mean()
                        avg_hatali_proba = df_res["Hatalı Olma Olasılığı"].mean()

                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Ortalama 'OK' Olasılığı", f"{avg_ok_proba:.2%}")
                        m_col2.metric("Ortalama 'Hatalı' Olasılığı", f"{avg_hatali_proba:.2%}")
        
        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}. Lütfen doğru model ve sütun dosyalarını yüklediğinizden emin olun.")
    else:
        st.warning("Lütfen devam etmek için yukarıdaki 3 dosyayı da yükleyin.")