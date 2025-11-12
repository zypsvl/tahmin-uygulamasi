# app.py

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import sys
from src.config import COLS_TO_EXCLUDE, PROGRAMDAKI_COLS, MEHMET_BEY_ATKI_COLS, MEHMET_BEY_COZGU_COLS, ALL_CSV_COLUMNS
from src.data_processing import load_and_prepare_base_data, normalize_columns, process_for_modeling
from src.model_training import train_and_evaluate_model

def get_data_folder_path(folder_name):
    """
    Program .exe olarak paketlendiğinde veya normal script olarak çalıştırıldığında
    veri klasörünün doğru yolunu bulur.
    """
    if hasattr(sys, "_MEIPASS"):   
        # PyInstaller geçici bir klasör oluşturur ve dosyaları oraya çıkarır.
        # sys._MEIPASS bu geçici klasörün yolunu içerir.
        return os.path.join(sys._MEIPASS, folder_name)
    
    # Normal bir .py scripti olarak çalıştırıldığında, mevcut dizini kullan.
    # Bu, geliştirme ortamında çalışmayı kolaylaştırır.
    return folder_name
# YENİ EKLENEN KISIM SONU


# Streamlit sayfa ayarları
st.set_page_config(layout="wide", page_title="Atkı-Çözgü Tahmin Modeli")
st.title("🧵 Atkı-Çözgü Çekme Değeri Tahmin Arayüzü")

# Session state yönetimi
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.base_data = None
    st.session_state.model = None
    st.session_state.model_cols = None
    st.session_state.target_col = None
    st.session_state.imputation_method = None

def read_data_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8-sig', header=0)
            except Exception:
                uploaded_file.seek(0); df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8-sig', header=None)
                if len(df.columns) == len(ALL_CSV_COLUMNS): df.columns = ALL_CSV_COLUMNS
                else:
                    st.error(f"Sütun sayısı uyuşmazlığı: Dosyada {len(df.columns)} sütun var, config'de {len(ALL_CSV_COLUMNS)} bekleniyordu.")
                    return None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Desteklenmeyen dosya formatı. Lütfen .csv veya .xlsx dosyası yükleyin."); return None
        return normalize_columns(df)
    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}"); return None

def align_columns(df, model_columns):
    missing_cols = set(model_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    return df[model_columns]

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Veri Yükleme")
    
    # DEĞİŞTİRİLEN KISIM BAŞLANGICI
    # Varsayılan yolu, programın çalışma şekline göre (normal vs .exe) dinamik olarak belirle
    default_folder_path = get_data_folder_path("T3verileri")
    folder_path = st.text_input("Eğitim Veri Klasörünün Yolu:", default_folder_path)
    # DEĞİŞTİRİLEN KISIM SONU
    
    if st.button("Verileri Yükle ve Hazırla"):
        with st.spinner("Veriler okunuyor ve hazırlanıyor..."):
            data, message = load_and_prepare_base_data(folder_path)
            if data is not None:
                st.session_state.base_data, st.session_state.data_loaded = data, True
                st.success(message)
            else:
                st.error(message); st.session_state.data_loaded = False

# --- ANA ARAYÜZ ---
if st.session_state.data_loaded:
    df_base = st.session_state.base_data
    with st.sidebar:
        st.header("2. Eğitim Parametreleri")
        
        # --- DEĞİŞİKLİK BAŞLANGICI: Filtreleme Seçeneği ---
        st.write("Eğitim Verisini Filtrele (Opsiyonel)")
        filtre_kriteri = st.radio(
            "Hangi özelliğe göre filtrelemek istersiniz?",
            ("Filtreleme Yok", "Tip Koduna Göre", "Karışıma Göre"),
            horizontal=True
        )

        secilen_deger = None
        if filtre_kriteri == "Tip Koduna Göre":
            tip_kodlari = ["Tüm Tipleri Kullan"] + sorted([str(tip) for tip in df_base["tip kodu"].unique() if pd.notna(tip)])
            secilen_deger = st.selectbox("Filtrelemek için 'tip kodu' seçin:", tip_kodlari)
        elif filtre_kriteri == "Karışıma Göre":
            if 'karisim' in df_base.columns:
                karisim_degerleri = ["Tüm Karışımları Kullan"] + sorted([str(k) for k in df_base["karisim"].unique() if pd.notna(k)])
                secilen_deger = st.selectbox("Filtrelemek için 'karışım' seçin:", karisim_degerleri)
            else:
                st.warning("'karisim' sütunu veride bulunamadı.")
                filtre_kriteri = "Filtreleme Yok"
        # --- DEĞİŞİKLİK SONU: Filtreleme Seçeneği ---

        hedef_degisken = st.radio("Hangi Değer Tahmin Edilecek?", ("Atkı Çekme (AtkıSınıf)", "Çözgü Çekme (CozguSınıf)"))
        algoritma = st.selectbox("Kullanılacak Algoritmayı Seçin:", ("Extra Trees Classifier", "Decision Tree Classifier"))
    
    st.subheader("Özellik Seti ve Veri İşleme Yöntemi")
    ozellik_secim_yontemi = st.radio(
        "Hangi yöntemi kullanmak istersiniz?",
        ("Programdaki Özellikler", 
         "Mehmet Bey'in Seçtiği Özellikler",
         "Manuel Seçim"),
        horizontal=True, 
        help="Programdaki özellikler için eksik veri içeren satırlar silinir. Mehmet Bey'in ve Manuel seçimde ise 0 ile doldurulur.")
    
    target_col = "AtkıSınıf" if "Atkı" in hedef_degisken else "CozguSınıf"
    base_feature_cols = sorted([col for col in df_base.columns if col not in COLS_TO_EXCLUDE and 'rolik numarası' not in col])
    
    if ozellik_secim_yontemi == "Programdaki Özellikler":
        selected_cols = [col for col in PROGRAMDAKI_COLS if col in base_feature_cols]
        imputation_method = 'drop'
    elif ozellik_secim_yontemi == "Mehmet Bey'in Seçtiği Özellikler":
        source_cols = MEHMET_BEY_ATKI_COLS if target_col == "AtkıSınıf" else MEHMET_BEY_COZGU_COLS
        selected_cols = [col for col in source_cols if col not in COLS_TO_EXCLUDE and col in base_feature_cols]
        imputation_method = 'fill_zero' # <-- DEĞİŞİKLİK BURADA YAPILDI
    else: # ozellik_secim_yontemi == "Manuel Seçim"
        excluded_cols = st.multiselect("Modelden Çıkarılacak Özellikler:", options=base_feature_cols)
        selected_cols = [col for col in base_feature_cols if col not in excluded_cols]
        imputation_method = 'fill_zero'

    st.markdown(f"Model için **{len(selected_cols)}** özellik seçildi.")
    st.markdown("---"); st.subheader("Model Eğitimi")
    if st.button("Eğitimi Başlat"):
        if not selected_cols: st.warning("Lütfen en az bir özellik seçin.")
        else:
            with st.spinner("Model eğitiliyor..."):
                data_for_training = df_base.copy()
                
                # --- DEĞİŞİKLİK BAŞLANGICI: Filtreleme Uygulaması ---
                if filtre_kriteri == "Tip Koduna Göre" and secilen_deger != "Tüm Tipleri Kullan":
                    data_for_training = data_for_training[data_for_training['tip kodu'].astype(str) == secilen_deger]
                elif filtre_kriteri == "Karışıma Göre" and secilen_deger != "Tüm Karışımları Kullan":
                    data_for_training = data_for_training[data_for_training['karisim'].astype(str) == secilen_deger]
                # --- DEĞİŞİKLİK SONU: Filtreleme Uygulaması ---
                
                X_train_ready, df_full = process_for_modeling(data_for_training, selected_cols, create_targets=True, fill_na_method=imputation_method)
                if X_train_ready.empty: st.error("Filtreleme ve ön işleme sonrası eğitim için veri kalmadı.")
                elif len(df_full[target_col].unique()) < 2: st.error(f"Eğitim için en az iki farklı sınıf (0 ve 1) gereklidir. Veride sadece '{df_full[target_col].unique()}' sınıfı bulundu.")
                else:
                    y_train_ready = df_full[target_col]
                    model, report, cm, acc, prec, rec, f1 = train_and_evaluate_model(X_train_ready, y_train_ready, algoritma)
                    st.session_state.update({'model': model, 'model_cols': list(X_train_ready.columns), 'target_col': target_col})
                    st.success("Model başarıyla eğitildi!")
                    st.subheader("--- EĞİTİM SETİ TEST SONUÇLARI ---")
                    c1,c2,c3,c4 = st.columns(4); c1.metric("Accuracy",f"{acc:.4f}"); c2.metric("Precision",f"{prec:.4f}"); c3.metric("Recall",f"{rec:.4f}"); c4.metric("F1-score",f"{f1:.4f}")
                    st.write("#### Detaylı Rapor"); st.dataframe(report.style.format("{:.4f}"))
                    st.write("#### Karışıklık Matrisi"); st.dataframe(cm)

# --- TEST BÖLÜMÜ (Bu bölümde değişiklik yok) ---
if st.session_state.model:
    st.markdown("---"); st.subheader("Eğitilmiş Modeli Test Et")
    
    st.write("#### Test Verisi Ön İşleme Yöntemi")
    test_imputation_choice = st.radio(
        "Test verisindeki eksik (boş) değerlere ne yapılsın?",
        ("Boş Değerleri 0 ile Doldur (Tüm satırlar için tahmin yapılır)", "Boş Değer İçeren Satırları Sil (Sadece tam veriler tahmin edilir)"),
        horizontal=True,
        help="0 ile doldurma, tüm satırlar için tahmin üretir. Satır silme ise sadece tam veriye sahip satırlar için tahmin yapar."
    )
    
    test_na_method = 'drop' if "Sil" in test_imputation_choice else 'fill_zero'

    tab1, tab2, tab3 = st.tabs(["🎯 Tahmin (Etiketsiz Veri)", "📊 Doğrulama (Etiketli Veri)", "💾 Modeli Dışa Aktar"])
    
    def handle_test_logic(df_raw, model, model_cols, create_targets, target_col, test_na_method):
        initial_rows = len(df_raw)
        if create_targets:
            X_ready, df_full = process_for_modeling(df_raw, model_cols, create_targets=True, fill_na_method=test_na_method)
        else:
            X_ready = process_for_modeling(df_raw, model_cols, create_targets=False, fill_na_method=test_na_method)
        if X_ready.empty:
            st.error("Ön işleme sonrası test edilecek veri kalmadı. Lütfen dosyanızı veya ön işleme yönteminizi kontrol edin."); return
        if test_na_method == 'drop':
            rows_dropped = initial_rows - len(X_ready)
            if rows_dropped > 0:
                st.info(f"**Bilgi:** Test dosyanızdaki eksik veri içeren **{rows_dropped}** satır analizden çıkarıldı.")
        X_aligned = align_columns(X_ready, model_cols)
        y_pred = model.predict(X_aligned)
        if create_targets:
            y_true = df_full.loc[X_aligned.index, target_col]
            st.write("#### Doğrulama Verisindeki Sınıf Dağılımı")
            class_distribution = y_true.value_counts()
            class_labels = {0: "Sınıf 0 (OK)", 1: "Sınıf 1 (Hatalı)"}
            class_distribution = class_distribution.rename(index=class_labels)
            st.dataframe(class_distribution)
            st.success(f"Doğrulama tamamlandı! {len(X_aligned)} satır üzerindeki performans:")
            acc=accuracy_score(y_true,y_pred); prec=precision_score(y_true,y_pred,zero_division=0); rec=recall_score(y_true,y_pred,zero_division=0); f1=f1_score(y_true,y_pred,zero_division=0)
            c1,c2,c3,c4 = st.columns(4); c1.metric("Accuracy",f"{acc:.4f}"); c2.metric("Precision",f"{prec:.4f}"); c3.metric("Recall",f"{rec:.4f}"); c4.metric("F1-score",f"{f1:.4f}")
        else:
            y_proba = model.predict_proba(X_aligned)
            df_res = pd.DataFrame({"Tahmin (0:OK, 1:Hatalı)": y_pred, "OK Olasılığı": y_proba[:,0], "Hatalı Olasılığı": y_proba[:,1]}, index=X_aligned.index)
            st.success(f"Tahminler {len(df_res)} satır için oluşturuldu."); 
            st.dataframe(df_res)
            st.subheader("Genel Ortalama Olasılıklar")
            st.info("Aşağıdaki metrikler, yukarıdaki tablodaki tüm satırların olasılıklarının ortalamasını göstermektedir.")
            avg_ok_proba = df_res["OK Olasılığı"].mean()
            avg_hatali_proba = df_res["Hatalı Olasılığı"].mean()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Ortalama 'OK' Olasılığı", value=f"{avg_ok_proba:.2%}")
            with col2:
                st.metric(label="Ortalama 'Hatalı' Olasılığı", value=f"{avg_hatali_proba:.2%}")

    with tab1:
        up_predict_file = st.file_uploader("Tahmin edilecek dosyayı yükleyin:", type=["csv","xlsx","xls"], key="up_predict")
        if st.button("Tahmin Et", key="btn_predict"):
            if up_predict_file:
                df_raw = read_data_file(up_predict_file)
                if df_raw is not None:
                    handle_test_logic(df_raw, st.session_state.model, st.session_state.model_cols, create_targets=False, target_col=None, test_na_method=test_na_method)
            else: st.warning("Lütfen tahmin için bir dosya yükleyin.")

    with tab2:
        up_val_file = st.file_uploader("Doğrulama yapılacak dosyayı yükleyin:", type=["csv","xlsx","xls"], key="up_validate")
        if st.button("Doğrulama Yap", key="btn_validate"):
            if up_val_file:
                df_raw = read_data_file(up_val_file)
                if df_raw is not None:
                    handle_test_logic(df_raw, st.session_state.model, st.session_state.model_cols, create_targets=True, target_col=st.session_state.target_col, test_na_method=test_na_method)
            else: st.warning("Lütfen doğrulama için bir dosya yükleyin.")

    with tab3:
        st.info("Eğitilmiş modeli ve kullandığı sütun listesini bilgisayarınıza indirin.")
        c1, c2 = st.columns(2)
        model_bytes = BytesIO(); joblib.dump(st.session_state.model, model_bytes); model_bytes.seek(0)
        c1.download_button(label="🤖 Modeli İndir (.pkl)", data=model_bytes, file_name="egitilmis_model.pkl")
        cols_bytes = BytesIO(); joblib.dump(st.session_state.model_cols, cols_bytes); cols_bytes.seek(0)
        c2.download_button(label="📊 Model Sütunlarını İndir (.pkl)", data=cols_bytes, file_name="model_sutunlari.pkl")

st.markdown("---"); st.header("Harici Model ile Test")
up_ext_model = st.file_uploader("1. Model (.pkl)", type="pkl", key="ext_model")
up_ext_cols = st.file_uploader("2. Model Sütunları (.pkl)", type="pkl", key="ext_cols")
up_ext_test = st.file_uploader("3. Test Dosyası (CSV/XLSX)", type=["csv","xlsx","xls"], key="ext_csv")

if st.button("Harici Model ile Test Et", key="btn_ext_test"):
    if up_ext_model and up_ext_cols and up_ext_test:
        df_raw = read_data_file(up_ext_test)
        if df_raw is not None:
            try:
                model = joblib.load(up_ext_model); model_cols = joblib.load(up_ext_cols)
                handle_test_logic(df_raw, model, model_cols, create_targets=False, target_col=None, test_na_method=test_na_method)
            except Exception as e: st.error(f"Harici test sırasında hata: {e}")
    else: st.warning("Lütfen 3 dosyayı da yükleyin.")