import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import copy
import h5netcdf
import os
import time
import re
from datetime import datetime, timedelta

# --- НАСТРОЙКИ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Ice Forecast NSR", layout="wide", page_icon="🚢")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    /* Увеличиваем заголовки графиков */
    .css-10trblm {font-size: 1.2rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Модель прогнозирования ледовой обстановки на СМП")
st.markdown("**Модуль верификации: Сравнение прогноза с фактическими данными спутникового мониторинга**")
st.markdown("---")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Ошибка модели: {e}")
    model = None

# --- ПАРСИНГ ДАТ ---
def extract_date(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except:
            return None
    return None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Данные и Настройки")
    
    if model is None:
        st.error("❌ Файл модели не найден!")
        st.stop()
    else:
        st.success("✅ Нейросеть активна")
    
    uploaded_files = st.file_uploader(
        "Загрузка массива данных (.nc)", 
        type=['nc'], 
        accept_multiple_files=True
    )
    
    file_db = {}
    if uploaded_files:
        for f in uploaded_files:
            dt = extract_date(f.name)
            if dt:
                file_db[dt] = f
        
        sorted_dates = sorted(file_db.keys())
        st.caption(f"Доступно снимков: {len(file_db)}")
        
        if len(file_db) > 0:
            st.markdown("---")
            
            # ВЫБОР ДАТЫ СТАРТА
            start_date = st.selectbox(
                "1. Дата старта расчета", 
                options=sorted_dates,
                format_func=lambda x: x.strftime("%d.%m.%Y")
            )
            
            # ВЫБОР ГОРИЗОНТА
            horizon = st.slider("2. Горизонт прогноза (суток)", 1, 14, 6)
            
            # ЦЕЛЕВАЯ ДАТА
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.info(f"📅 Прогноз на: **{target_date.strftime('%d.%m.%Y')}**")
            
            if has_truth:
                st.caption("✅ Контрольный файл найден")
                btn_disabled = False
            else:
                st.warning("⚠️ Нет файла для проверки")
                btn_disabled = True
            
            predict_btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary", disabled=btn_disabled)

# --- ОСНОВНАЯ ЛОГИКА ---
if 'predict_btn' in locals() and predict_btn:
    try:
        status_container = st.status("Инициализация математической модели...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТА
        start_file_obj = file_db[start_date]
        start_file_obj.seek(0)
        with open("start_temp.nc", "wb") as f: f.write(start_file_obj.read())
        
        ds = xr.open_dataset("start_temp.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        
        def clean(d):
            d = np.nan_to_num(d, nan=0.0)
            d = np.where(d > 100, 0, d)
            if np.max(d) > 1.05: d = d / 100.0
            return d

        current_img = clean(data_raw)
        
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ЦИКЛ ПРОГНОЗА (С ИНЕРЦИЕЙ - STABILIZATION)
        progress_bar = status_container.progress(0)
        alpha = 0.75 # Коэффициент сохранения структуры (Инерция)
        
        for day in range(1, horizon + 1):
            # Прогноз нейросети
            pred_ai = model.predict(input_batch, verbose=0)
            
            # Стабилизация (смешиваем с предыдущим шагом)
            # Это убирает эффект "размытого пятна"
            pred_stabilized = (input_batch * alpha) + (pred_ai * (1 - alpha))
            
            # Фильтрация шума
            pred_clean = tf.where(pred_stabilized > 0.1, pred_stabilized, 0.0)
            
            input_batch = pred_clean 
            
            sim_date = start_date + timedelta(days=day)
            status_container.write(f"✅ Шаг {day}/{horizon}: Расчет дрейфа на {sim_date.strftime('%d.%m')}")
            progress_bar.progress(day / horizon)
            time.sleep(0.05)
            
        # 3. ПОСТ-ОБРАБОТКА
        final_small = input_batch[0]
        final_full = tf.image.resize(final_small, [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        
        final_viz = copy.deepcopy(final_full)
        final_viz[land
