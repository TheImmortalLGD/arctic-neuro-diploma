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
import glob
from datetime import datetime, timedelta

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Ice Forecast NSR", layout="wide", page_icon="🚢")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Модель прогнозирования ледовой обстановки на СМП")
st.markdown("**Модуль верификации и анализа ошибок**")
st.markdown("---")

# --- МОДЕЛЬ ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except:
    model = None

# --- УТИЛИТЫ ---
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
    st.header("🗂️ Данные")
    
    if model is None:
        st.error("❌ Нет файла модели (.h5)")
        st.stop()
    
    # Загрузка
    uploaded_files = st.file_uploader(
        "Загрузите файлы (.nc)", 
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
        st.info(f"Загружено снимков: {len(file_db)}")
        
        if len(file_db) > 0:
            st.markdown("---")
            start_date = st.selectbox("Дата старта", options=sorted_dates, format_func=lambda x: x.strftime("%d.%m.%Y"))
            horizon = st.slider("Горизонт (сут.)", 1, 7, 3)
            
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.write(f"Цель: **{target_date.strftime('%d.%m.%Y')}**")
            
            if not has_truth:
                st.warning("Нет файла для проверки")
                btn = False
            else:
                btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary")

# --- ЛОГИКА ---
if 'btn' in locals() and btn:
    try:
        status = st.status("Нейросетевое моделирование...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТА
        f_obj = file_db[start_date]
        f_obj.seek(0)
        with open("temp_start.nc", "wb") as f: f.write(f_obj.read())
        
        ds = xr.open_dataset("temp_start.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Маски и очистка
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        
        def clean(d):
            d = np.nan_to_num(d, nan=0.0)
            d = np.where(d > 100, 0, d)
            if np.max(d) > 1.05: d = d / 100.0
            return d

        current_img = clean(data_raw)
        
        # Тензор
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ЦИКЛ ПРОГНОЗА
        prog_bar = status.progress(0)
        alpha = 0.75
        
        for day in range(1, horizon + 1):
            pred_ai = model.predict(input_batch, verbose=0)
            pred_stab = (input_batch * alpha) + (pred_ai * (1 - alpha))
            pred_clean = tf.where(pred_stab > 0.1, pred_stab, 0.0)
            input_batch = pred_clean
            
            status.write(f"✅ День {day}: Расчет завершен")
            prog_bar.progress(day / horizon)
        
        # Результат прогноза
        final_full = tf.image.resize(input_batch[0], [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status.update(label="Готово", state="complete", expanded=False)

        # 3. ПОДГОТОВКА ФАКТА
        t_obj = file_db[target_date]
        t_obj.seek(0)
        with open("temp_target.nc", "wb") as f: f.write(t_obj.read())
        
        ds_t = xr.open_dataset("temp_target.nc", engine='h5netcdf')
        target_raw = ds_t[var_name].isel(time=0).squeeze().values
        target_clean = clean(target_raw)
        
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # 4. РАСЧЕТ ОШИБКИ (Diff Map)
        # Считаем абсолютную разницу
        diff_map = np.abs(final_full - target_clean)
        diff_map[land_mask] = np.nan # Убираем сушу с карты ошибок
        
        mae = np.nanmean(diff_map) * 100
        acc = 100 - mae

        # 5. ВИЗУАЛИЗАЦИЯ (ТРИ КОЛОНКИ)
        st.subheader(f"📊 Анализ результатов (Горизонт: {horizon} сут.)")
        
        c1, c2, c3 = st.columns(3)
        cmap_ice = plt.cm.Blues_r.copy()
        cmap_ice.set_bad('#1E1E1E')
        
        # Колонка 1: Прогноз
        with c1:
            st.markdown("### 🧠 Прогноз ИИ")
            fig1, ax1 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax1.imshow(final_viz, cmap=cmap_ice, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        # Колонка 2: Факт
        with c2:
            st.markdown("### 🛰️ Факт (Спутник)")
            fig2, ax2 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax2.imshow(target_viz, cmap=cmap_ice, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
            
        # Колонка 3: Карта ошибок (НОВОЕ)
        with c3:
            st.markdown("### 🔥 Карта ошибок")
            fig3, ax3 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            # Используем карту 'hot' (черный -> красный -> желтый)
            # vmax=0.5 означает, что ошибка в 50% будет светиться максимально ярко
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=0.5) 
            ax3.axis('off')
            # Добавляем шкалу (colorbar)
            cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white') # Белые цифры шкалы
            st.pyplot(fig3)
            
        # Метрики
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность (Accuracy)", f"{acc:.2f}%")
        m2.metric("Ср. ошибка (MAE)", f"{mae:.2f}%")
        m3.metric("Статус", "✅ УСПЕХ" if acc > 80 else "⚠️ ТРЕБУЕТ ВНИМАНИЯ")

    except Exception as e:
        st.error(f"Ошибка: {e}")

elif not uploaded_files:
    st.info("👈 Загрузите файлы .nc для начала.")
