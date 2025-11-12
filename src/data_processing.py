# src/data_processing.py

import os
import re
import pandas as pd
import numpy as np
import streamlit as st

def normalize_columns(df):
    """Sütun isimlerindeki fazla boşlukları temizler."""
    df.columns = [re.sub(r'\s+', ' ', col).strip() if isinstance(col, str) else col for col in df.columns]
    return df

def parse_range(val):
    """'10-20', '10;20' gibi metinleri (min, max) tuple'ına çevirir."""
    if not isinstance(val, str): return None
    s = val.strip().replace(",", ".").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\d\-\.\;\:\s]+", " ", s)
    s = re.sub(r"[\:\|/]", ";", s); s = re.sub(r"\s*;\s*", ";", s)
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", s)
    if len(nums) < 2: return None
    a, b = float(nums[0]), float(nums[1])
    return (a, b) if a <= b else (b, a)

def safe_float(x):
    """Bir değeri güvenli bir şekilde float'a dönüştürür."""
    try:
        if pd.isna(x): return np.nan
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError): return np.nan

def classify_atki_helper(row):
    """Atkı çekme değeri için sınıflandırma yapar (0: OK, 1: Hatalı)."""
    rng = parse_range(row.get("AtkiCekme_Standart"))
    x = safe_float(row.get("AtkiCekme_Test"))
    if rng is None or pd.isna(x): return 1
    low, high = rng
    return 0 if (low <= x <= high) else 1

def classify_cozgu_helper(row):
    """Çözgü çekme değeri için sınıflandırma yapar (0: OK, 1: Hatalı)."""
    rng = parse_range(row.get("CozguCekme_Standart"))
    x = safe_float(row.get("CozguCekme_Test"))
    if rng is None or pd.isna(x): return 1
    low, high = rng
    return 0 if (low <= x <= high) else 1

def process_for_modeling(df, feature_cols, create_targets=False, fill_na_method='drop'):
    """
    Veriyi modelleme için hazırlar.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    processed_df = df.copy()

    if create_targets:
        required_target_cols = ["AtkiCekme_Standart", "AtkiCekme_Test", "CozguCekme_Standart", "CozguCekme_Test"]
        if set(required_target_cols).issubset(processed_df.columns):
            processed_df["AtkıSınıf"] = processed_df.apply(classify_atki_helper, axis=1)
            processed_df["CozguSınıf"] = processed_df.apply(classify_cozgu_helper, axis=1)

    final_feature_cols = [col for col in feature_cols if col in processed_df.columns]
    for col in final_feature_cols:
        processed_df[col] = pd.to_numeric(
            processed_df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

    if fill_na_method == 'drop':
        processed_df.dropna(subset=final_feature_cols, how='any', inplace=True)
    elif fill_na_method == 'fill_zero':
        processed_df[final_feature_cols] = processed_df[final_feature_cols].fillna(0)

    if "Tarih_Saat" in processed_df.columns:
        processed_df["Tarih_Saat"] = pd.to_datetime(processed_df["Tarih_Saat"], errors="coerce")
        processed_df.dropna(subset=["Tarih_Saat"], inplace=True)
        processed_df = processed_df.set_index('Tarih_Saat')

    if processed_df.empty:
        return pd.DataFrame()

    X_final = processed_df[[col for col in final_feature_cols if col in processed_df.columns]]
    return X_final