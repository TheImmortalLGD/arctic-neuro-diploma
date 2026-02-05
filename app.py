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
    </style>
    """, unsafe_allow_html=True)

# === НОВЫЙ ЗАГОЛОВОК ===
st.title("🚢 Модель для прогнозирования ледовой обстановки на СМП")
st.markdown("**Автоматизированная система краткосрочного и среднесрочного прогнозирования**")
st.info("ℹ️ Система работает в режиме валидации на исторических данных (Backtesting).")
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

# --- ФУНКЦИЯ ПОИСКА ДАТЫ ---
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
    st.header("🗂️ Панель управления")
    
    if model is None:
        st.error("❌ Файл модели не найден!")
        st.stop()
    else:
        st.success("✅ Нейросеть активна")
    
    # === ИСПРАВЛЕННОЕ НАЗВАНИЕ ===
    uploaded_files = st.file_uploader(
        "Загрузка массива спутниковых данных (.nc)", 
        type=['nc'], 
        accept_multiple_files=True
    )
    
    # Индексация
    file_db = {}
    if uploaded_files:
        for f in uploaded_files:
            dt = extract_date(f.name)
            if dt:
                file_db[dt] = f
        
        sorted_dates = sorted(file_db.keys())
        st.caption(f"В систему загружено снимков: {len(file_db)}")
        
        if len(file_db) > 0:
            st.markdown("---")
            st.header("⚙️ Параметры прогноза")
            
            # ВЫБОР ДАТЫ СТАРТА
            start_date = st.selectbox(
                "1. Дата начала моделирования", 
                options=sorted_dates,
                format_func=lambda x: x.strftime("%d.%m.%Y")
            )
            
            # ВЫБОР ГОРИЗОНТА
            max_horizon = 14
            horizon = st.slider("2. Горизонт прогноза (суток)", 1, max_horizon, 3)
            
            # ЦЕЛЕВАЯ ДАТА
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.markdown(f"**Дата прогноза:** `{target_date.strftime('%d.%m.%Y')}`")
            
            if has_truth:
                st.info("✅ Контрольный снимок найден")
                btn_disabled = False
            else:
                st.warning("⚠️ Нет данных для сверки на эту дату")
                btn_disabled = True
            
            predict_btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary", disabled=btn_disabled)

# --- ОСНОВНАЯ ЛОГИКА ---
if 'predict_btn' in locals() and predict_btn:
    try:
        status_container = st.status("Выполнение нейросетевого моделирования...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТОВОГО ФАЙЛА
        start_file_obj = file_db[start_date]
        start_file_obj.seek(0)
        
        with open("start_temp.nc", "wb") as f: f.write(start_file_obj.read())
        
        ds = xr.open_dataset("start_temp.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Подготовка
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
        
        # 2. ЦИКЛ ПРОГНОЗА
        progress_bar = status_container.progress(0)
        
        for day in range(1, horizon + 1):
            pred = model.predict(input_batch, verbose=0)
            input_batch = pred 
            
            sim_date = start_date + timedelta(days=day)
            status_container.write(f"✅ Расчет на {sim_date.strftime('%d.%m.%Y')} завершен")
            progress_bar.progress(day / horizon)
            time.sleep(0.1)
            
        # 3. ПОСТ-ОБРАБОТКА
        final_small = input_batch[0]
        final_full = tf.image.resize(final_small, [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status_container.update(label="Моделирование завершено успешно", state="complete", expanded=False)

        # 4. СРАВНЕНИЕ И ВИЗУАЛИЗАЦИЯ
        target_file_obj = file_db[target_date]
        target_file_obj.seek(0)
        with open("target_temp.nc", "wb") as f: f.write(target_file_obj.read())
        
        ds_target = xr.open_dataset("target_temp.nc", engine='h5netcdf')
        target_raw = ds_target[var_name].isel(time=0).squeeze().values
        target_clean = clean(target_raw)
        
        # Расчет ошибки
        diff = np.abs(final_full - target_clean)
        diff[land_mask] = np.nan
        mae = np.nanmean(diff) * 100
        accuracy = 100 - mae
        
        st.subheader(f"📊 Отчет: Прогноз на {horizon} сут. ({target_date.strftime('%d.%m.%Y')})")
        
        col1, col2, col3 = st.columns(3)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E')
        
        with col1:
            st.caption("Исходная ледовая обстановка")
            fig1, ax1 = plt.subplots(facecolor='#0e1117')
            start_viz = copy.deepcopy(current_img)
            start_viz[land_mask] = np.nan
            ax1.imshow(start_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.caption("Результат моделирования (AI)")
            fig2, ax2 = plt.subplots(facecolor='#0e1117')
            ax2.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
            
        with col3:
            st.caption("Фактические данные (Спутник)")
            fig3, ax3 = plt.subplots(facecolor='#0e1117')
            target_viz = copy.deepcopy(target_clean)
            target_viz[land_mask] = np.nan
            ax3.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax3.axis('off')
            st.pyplot(fig3)
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность прогноза", f"{accuracy:.2f}%")
        m2.metric("Горизонт", f"{horizon} сут.")
        m3.metric("Валидация", "УСПЕХ" if accuracy > 80 else "ОТКЛОНЕНИЕ", delta="Pass" if accuracy > 80 else "-Warn")
        
        with st.expander("🔎 Открыть карту отклонений"):
            fig_err, ax_err = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            diff_viz = copy.deepcopy(diff)
            im = ax_err.imshow(diff_viz, cmap='hot', vmin=0, vmax=0.4)
            plt.colorbar(im, ax=ax_err, label="Ошибка концентрации")
            ax_err.set_title("Зоны расхождения прогноза с фактом", color='white')
            ax_err.axis('off')
            st.pyplot(fig_err)

    except Exception as e:
        st.error(f"Ошибка выполнения: {e}")

elif not uploaded_files:
    st.info("👈 Загрузите архив данных (.nc) в меню слева для начала работы.")
