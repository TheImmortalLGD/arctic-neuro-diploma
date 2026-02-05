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

# --- НАСТРОЙКИ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Ice Forecast NSR", layout="wide", page_icon="🚢")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Модель прогнозирования ледовой обстановки на СМП")
st.markdown("**Система поддержки принятия решений (СППР)**")
st.markdown("---")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_ai_model():
    # Проверяем наличие файла модели
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    model = None

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def extract_date(filename):
    """Извлекает дату из имени файла (например, ...20200401.nc -> 01.04.2020)"""
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except:
            return None
    return None

def clean_data(d):
    """Очистка данных: убирает NaN, маскирует сушу, нормализует"""
    d = np.nan_to_num(d, nan=0.0)
    d = np.where(d > 100, 0, d)
    if np.max(d) > 1.05: d = d / 100.0
    return d

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Данные для анализа")
    
    if model is None:
        st.error("❌ Файл модели (ice_model_month_v2.h5) не найден.")
        st.stop()
    else:
        st.success("✅ Нейросеть активна")

    # ЗАГРУЗКА ФАЙЛОВ ВРУЧНУЮ (Самый надежный способ)
    uploaded_files = st.file_uploader(
        "Загрузите файлы .nc (Апрель)", 
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
        st.info(f"В систему загружено: {len(file_db)} снимков")
        
        if len(file_db) > 0:
            st.markdown("---")
            start_date = st.selectbox("Дата старта", options=sorted_dates, format_func=lambda x: x.strftime("%d.%m.%Y"))
            horizon = st.slider("Горизонт прогноза (сут.)", 1, 7, 3)
            
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.write(f"Целевая дата: **{target_date.strftime('%d.%m.%Y')}**")
            
            if not has_truth:
                st.warning("⚠️ Нет файла для проверки прогноза")
                btn = False
            else:
                btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary")

# --- ОСНОВНАЯ ЛОГИКА ---
if 'btn' in locals() and btn:
    try:
        status = st.status("Инициализация вычислительного ядра...", expanded=True)
        
        # === ФУНКЦИЯ БЕЗОПАСНОГО ЧТЕНИЯ (С ЗАЩИТОЙ ОТ ОШИБОК) ===
        def safe_open_nc(file_obj, temp_name):
            # 1. Сохраняем во временный файл
            file_obj.seek(0)
            with open(temp_name, "wb") as f:
                f.write(file_obj.read())
            
            # 2. ПРОВЕРКА РАЗМЕРА (Защита от GitHub LFS ссылок)
            size = os.path.getsize(temp_name)
            if size < 2000: # Если меньше 2 Кбайт
                st.error(f"❌ Критическая ошибка: Файл {file_obj.name} поврежден или является ссылкой.")
                st.warning("Размер файла слишком мал (< 2 Кб). Пожалуйста, загрузите ОРИГИНАЛЬНЫЙ файл с вашего компьютера (размером > 5 Мб).")
                st.stop()
            
            # 3. ПОПЫТКА ОТКРЫТИЯ (Перебор движков)
            engines = ['netcdf4', 'h5netcdf', 'scipy', None]
            
            for engine in engines:
                try:
                    ds = xr.open_dataset(temp_name, engine=engine)
                    return ds
                except:
                    continue
            
            raise ValueError("Не удалось открыть файл. Проверьте формат NetCDF.")

        # 1. ЧТЕНИЕ СТАРТОВОГО СНИМКА
        ds = safe_open_nc(file_db[start_date], "temp_start.nc")
        
        # Авто-поиск переменной льда
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Подготовка масок
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        current_img = clean_data(data_raw)
        
        # Тензор для нейросети
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ЦИКЛ ПРОГНОЗИРОВАНИЯ
        prog_bar = status.progress(0)
        alpha = 0.75 # Коэффициент инерции (Стабилизатор)
        
        for day in range(1, horizon + 1):
            # Прогноз ИИ
            pred_ai = model.predict(input_batch, verbose=0)
            
            # Стабилизация (смешивание с предыдущим шагом)
            pred_stab = (input_batch * alpha) + (pred_ai * (1 - alpha))
            
            # Фильтрация шума
            pred_clean = tf.where(pred_stab > 0.1, pred_stab, 0.0)
            
            # Обновление входа
            input_batch = pred_clean
            
            status.write(f"✅ День {day}: Моделирование дрейфа завершено")
            prog_bar.progress(day / horizon)
        
        # Восстановление размера
        final_full = tf.image.resize(input_batch[0], [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status.update(label="Расчет успешно завершен", state="complete", expanded=False)

        # 3. ЧТЕНИЕ ФАКТА (TARGET)
        ds_t = safe_open_nc(file_db[target_date], "temp_target.nc")
        target_raw = ds_t[var_name].isel(time=0).squeeze().values
        target_clean = clean_data(target_raw)
        
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # 4. РАСЧЕТ МЕТРИК И ОШИБОК
        diff_map = np.abs(final_full - target_clean)
        diff_map[land_mask] = np.nan # Игнорируем сушу
        
        mae = np.nanmean(diff_map) * 100
        accuracy = 100 - mae

        # 5. ВИЗУАЛИЗАЦИЯ (3 КОЛОНКИ)
        st.subheader(f"📊 Результаты валидации (Горизонт: {horizon} сут.)")
        
        c1, c2, c3 = st.columns(3)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E')
        
        with c1:
            st.markdown("### 🧠 Прогноз ИИ")
            fig1, ax1 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax1.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        with c2:
            st.markdown("### 🛰️ Факт")
            fig2, ax2 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax2.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
            
        with c3:
            st.markdown("### 🔥 Карта ошибок")
            fig3, ax3 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            # Тепловая карта ошибок (от 0 до 50%)
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            ax3.axis('off')
            st.pyplot(fig3)
            
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность прогноза", f"{accuracy:.2f}%")
        m2.metric("Средняя ошибка (MAE)", f"{mae:.2f}%")
        m3.metric("Статус теста", "УСПЕХ" if accuracy > 80 else "ТРЕБУЕТ КАЛИБРОВКИ")

    except Exception as e:
        st.error(f"Системная ошибка: {e}")

elif not uploaded_files:
    st.info("👈 Пожалуйста, загрузите .nc файлы (апрель) в меню слева.")
