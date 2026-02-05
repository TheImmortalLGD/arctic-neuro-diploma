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

# --- НАСТРОЙКИ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Ice Forecast NSR", layout="wide", page_icon="🚢")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    .css-10trblm {font-size: 1.2rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Модель прогнозирования ледовой обстановки на СМП")
st.markdown("**Модуль верификации: Автоматический анализ архива данных**")
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

# --- ФУНКЦИЯ ЧТЕНИЯ (Универсальная) ---
def read_nc_file(file_obj):
    """Читает файл и возвращает байты, независимо от того, загружен он или лежит локально"""
    if isinstance(file_obj, str): # Если это путь к файлу (на GitHub)
        with open(file_obj, "rb") as f:
            return f.read()
    else: # Если это загруженный файл (UploadedFile)
        file_obj.seek(0)
        return file_obj.read()

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Источники данных")
    
    if model is None:
        st.error("❌ Файл модели не найден!")
        st.stop()
    else:
        st.success("✅ Нейросеть активна")
    
    # 1. АВТО-ПОИСК ФАЙЛОВ В РЕПОЗИТОРИИ
    local_files = glob.glob("*.nc") # Ищем все .nc файлы рядом со скриптом
    # Исключаем временные файлы, которые мы сами создаем
    local_files = [f for f in local_files if "temp" not in f]
    
    file_db = {}
    
    # Добавляем локальные файлы (с GitHub)
    for f_path in local_files:
        dt = extract_date(f_path)
        if dt:
            file_db[dt] = f_path # Сохраняем путь как строку
            
    # 2. РУЧНАЯ ЗАГРУЗКА (Если нужно добавить что-то еще)
    uploaded_files = st.file_uploader(
        "Добавить файлы вручную (опционально)", 
        type=['nc'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for f in uploaded_files:
            dt = extract_date(f.name)
            if dt:
                file_db[dt] = f # Сохраняем объект файла
    
    # ИТОГОВАЯ ИНФОРМАЦИЯ
    sorted_dates = sorted(file_db.keys())
    count_local = len([x for x in file_db.values() if isinstance(x, str)])
    count_upload = len(file_db) - count_local
    
    st.caption(f"📚 В базе: {len(file_db)} снимков")
    if count_local > 0:
        st.caption(f"• Из репозитория: {count_local}")
    if count_upload > 0:
        st.caption(f"• Загружено вручную: {count_upload}")

    if len(file_db) > 0:
        st.markdown("---")
        
        # ВЫБОР ДАТЫ СТАРТА
        start_date = st.selectbox(
            "1. Дата старта", 
            options=sorted_dates,
            format_func=lambda x: x.strftime("%d.%m.%Y")
        )
        
        # ВЫБОР ГОРИЗОНТА
        # Ограничиваем, чтобы не выйти за пределы имеющихся файлов
        horizon = st.slider("2. Горизонт прогноза (суток)", 1, 7, 3)
        
        # ЦЕЛЕВАЯ ДАТА
        target_date = start_date + timedelta(days=horizon)
        has_truth = target_date in file_db
        
        st.info(f"📅 Прогноз на: **{target_date.strftime('%d.%m.%Y')}**")
        
        if has_truth:
            st.caption("✅ Данные для проверки есть")
            btn_disabled = False
        else:
            st.warning(f"⚠️ Нет файла за {target_date.strftime('%d.%m')}")
            btn_disabled = True
        
        predict_btn = st.button("🚀 ЗАПУСТИТЬ", type="primary", disabled=btn_disabled)

# --- ОСНОВНАЯ ЛОГИКА ---
if 'predict_btn' in locals() and predict_btn:
    try:
        status_container = st.status("Инициализация ядра моделирования...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТА
        file_content = read_nc_file(file_db[start_date])
        with open("start_temp.nc", "wb") as f: f.write(file_content)
        
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
        
        # 2. ЦИКЛ ПРОГНОЗА (С ИНЕРЦИЕЙ)
        progress_bar = status_container.progress(0)
        alpha = 0.75 
        
        for day in range(1, horizon + 1):
            pred_ai = model.predict(input_batch, verbose=0)
            pred_stabilized = (input_batch * alpha) + (pred_ai * (1 - alpha))
            pred_clean = tf.where(pred_stabilized > 0.1, pred_stabilized, 0.0)
            input_batch = pred_clean 
            
            sim_date = start_date + timedelta(days=day)
            status_container.write(f"✅ Расчет: {sim_date.strftime('%d.%m.%Y')}")
            progress_bar.progress(day / horizon)
            time.sleep(0.05)
            
        # 3. ПОСТ-ОБРАБОТКА
        final_small = input_batch[0]
        final_full = tf.image.resize(final_small, [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status_container.update(label="Моделирование завершено", state="complete", expanded=False)

        # 4. ПОДГОТОВКА ФАКТА
        target_content = read_nc_file(file_db[target_date])
        with open("target_temp.nc", "wb") as f: f.write(target_content)
        
        ds_target = xr.open_dataset("target_temp.nc", engine='h5netcdf')
        target_raw = ds_target[var_name].isel(time=0).squeeze().values
        target_clean = clean(target_raw)
        
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # Метрики
        diff = np.abs(final_full - target_clean)
        diff[land_mask] = np.nan
        mae = np.nanmean(diff) * 100
        accuracy = 100 - mae

        # 5. ВИЗУАЛИЗАЦИЯ
        st.subheader(f"📊 Результаты: Прогноз на {horizon} сут.")
        
        col1, col2 = st.columns(2)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E')
        
        with col1:
            st.markdown(f"### 🧠 Прогноз ИИ")
            st.caption(f"Дата: {target_date.strftime('%d.%m.%Y')}")
            fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor='#0e1117')
            ax1.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.markdown(f"### 🛰️ Факт (Спутник)")
            st.caption(f"Дата: {target_date.strftime('%d.%m.%Y')}")
            fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor='#0e1117')
            ax2.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность (Accuracy)", f"{accuracy:.2f}%")
        m2.metric("MAE", f"{mae:.2f}%")
        m3.metric("Результат", "УСПЕХ" if accuracy > 75 else "НИЖЕ НОРМЫ", delta="OK" if accuracy > 75 else "Warn")
        
        with st.expander("🔎 Карта ошибок"):
            fig_err, ax_err = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            diff_viz = copy.deepcopy(diff)
            im = ax_err.imshow(diff_viz, cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar(im, ax=ax_err, label="Ошибка")
            ax_err.axis('off')
            st.pyplot(fig_err)

    except Exception as e:
        st.error(f"Ошибка выполнения: {e}")

elif len(file_db) == 0:
    st.info("В репозитории нет файлов .nc. Загрузите их на GitHub или перетащите сюда вручную.")
