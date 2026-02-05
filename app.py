import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import copy
import os
import time
import re
from datetime import datetime, timedelta

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Ice Forecast NSR", layout="wide", page_icon="🚢")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Модель прогнозирования ледовой обстановки на СМП")
st.markdown("---")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except:
    model = None

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def extract_date(filename):
    """Парсинг даты из имени файла"""
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except:
            return None
    return None

def preprocess_raw_data(data):
    """Первичная обработка сырых данных со спутника"""
    d = np.nan_to_num(data, nan=0.0)
    d = np.where(d > 100, 0, d) # Маскируем сушу (если она > 100)
    if np.max(d) > 1.05: 
        d = d / 100.0 # Нормализация в 0..1
    return d

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Входные данные")
    
    if model is None:
        st.error("❌ Файл модели не найден")
        st.stop()
    else:
        st.success("✅ Ядро модели активно")

    # Загрузчик файлов
    uploaded_files = st.file_uploader(
        "Загрузите массив данных (.nc)", 
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
        st.info(f"Индексировано снимков: {len(file_db)}")
        
        if len(file_db) > 0:
            st.markdown("---")
            start_date = st.selectbox("Дата старта", options=sorted_dates, format_func=lambda x: x.strftime("%d.%m.%Y"))
            horizon = st.slider("Горизонт прогноза (сут.)", 1, 7, 3)
            
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.write(f"Целевая дата: **{target_date.strftime('%d.%m.%Y')}**")
            
            if has_truth:
                btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary")
            else:
                st.warning("⚠️ Нет файла для верификации прогноза")
                btn = False

# --- ОСНОВНАЯ ЛОГИКА ---
if 'btn' in locals() and btn:
    try:
        # Индикатор прогресса
        status = st.status("Выполнение нейросетевого прогноза...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТОВОГО КАДРА
        f_obj = file_db[start_date]
        f_obj.seek(0)
        with open("temp_start.nc", "wb") as f: f.write(f_obj.read())
        
        # Открываем универсально (xarray сам разберется с форматом)
        ds = xr.open_dataset("temp_start.nc")
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Сохраняем маску суши для финала
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        
        # Препроцессинг
        current_img = preprocess_raw_data(data_raw)
        
        # Подготовка тензора (Input)
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ЦИКЛ ПРОГНОЗИРОВАНИЯ (РЕКУРСИЯ)
        prog_bar = status.progress(0)
        alpha = 0.75 # Коэффициент инерции (стабилизация картинки)
        
        for day in range(1, horizon + 1):
            # Шаг 1: Предсказание нейросети
            pred_ai = model.predict(input_batch, verbose=0)
            
            # Шаг 2: Стабилизация (смешивание с предыдущим кадром)
            # Это убирает "дребезг" и размытие
            pred_stab = (input_batch * alpha) + (pred_ai * (1 - alpha))
            
            # Шаг 3: Фильтрация шума
            pred_clean = tf.where(pred_stab > 0.1, pred_stab, 0.0)
            
            # Выход становится входом для следующего дня
            input_batch = pred_clean
            
            status.write(f"✅ День {day}: Расчет дрейфа завершен")
            prog_bar.progress(day / horizon)
        
        # 3. ВОССТАНОВЛЕНИЕ РЕЗУЛЬТАТА
        final_full = tf.image.resize(input_batch[0], [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan # Возвращаем сушу
        
        status.update(label="Моделирование завершено успешно", state="complete", expanded=False)

        # 4. ПОДГОТОВКА ЭТАЛОНА (Ground Truth)
        t_obj = file_db[target_date]
        t_obj.seek(0)
        with open("temp_target.nc", "wb") as f: f.write(t_obj.read())
        
        ds_t = xr.open_dataset("temp_target.nc")
        target_raw = ds_t[var_name].isel(time=0).squeeze().values
        
        # Очищаем эталон той же функцией
        target_clean = preprocess_raw_data(target_raw)
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # 5. РАСЧЕТ МЕТРИК И ОШИБОК
        diff_map = np.abs(final_full - target_clean)
        diff_map[land_mask] = np.nan # Не считаем ошибку на суше
        
        mae = np.nanmean(diff_map) * 100
        accuracy = 100 - mae

        # 6. ВИЗУАЛИЗАЦИЯ (ТРИПТИХ)
        st.subheader(f"📊 Результаты верификации (Горизонт: {horizon} сут.)")
        
        col1, col2, col3 = st.columns(3)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E') # Цвет суши
        
        with col1:
            st.markdown("### 🧠 Прогноз ИИ")
            fig1, ax1 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax1.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.markdown("### 🛰️ Факт (Спутник)")
            fig2, ax2 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax2.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
            
        with col3:
            st.markdown("### 🔥 Карта ошибок")
            fig3, ax3 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            # Тепловая карта: Черный -> Красный -> Желтый
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
            cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white')
            ax3.axis('off')
            st.pyplot(fig3)
        
        # ИТОГОВЫЕ МЕТРИКИ
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность (Accuracy)", f"{accuracy:.2f}%")
        m2.metric("Средняя ошибка (MAE)", f"{mae:.2f}%")
        m3.metric("Вердикт", "✅ ДОСТОВЕРНО" if accuracy > 80 else "⚠️ ТРЕБУЕТ УТОЧНЕНИЯ")

    except Exception as e:
        st.error(f"Произошла ошибка при расчете: {e}")

elif not uploaded_files:
    st.info("👈 Пожалуйста, загрузите файлы данных (.nc) через меню слева.")

