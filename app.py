import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import copy
import os
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

def preprocess_raw_data(data):
    d = np.nan_to_num(data, nan=0.0)
    d = np.where(d > 100, 0, d)
    if np.max(d) > 1.05: d = d / 100.0
    return d

# --- ЗАГРУЗКА ДАННЫХ (ГИБРИДНАЯ) ---
def get_data_files():
    file_db = {}
    
    # 1. АВТО-ПОИСК (GitHub Repo)
    # Ищем файлы, начинающиеся на 'fixed_' (наши исправленные файлы)
    local_files = glob.glob("fixed_*.nc")
    
    if len(local_files) > 0:
        source_type = "auto"
        for f_path in local_files:
            dt = extract_date(f_path)
            if dt:
                file_db[dt] = f_path # Сохраняем путь (строку)
    else:
        source_type = "manual"
    
    return file_db, source_type

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Входные данные")
    
    if model is None:
        st.error("❌ Модель не найдена")
        st.stop()
    else:
        st.success("✅ Ядро модели активно")

    # Получаем файлы (Авто или Ручные)
    file_db, source_type = get_data_files()
    
    # Если на GitHub пусто, просим загрузить вручную
    if len(file_db) == 0:
        uploaded_files = st.file_uploader("Загрузите файлы (.nc)", type=['nc'], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                dt = extract_date(f.name)
                if dt:
                    file_db[dt] = f # Сохраняем объект файла
            source_type = "manual"

    # СТАТИСТИКА
    sorted_dates = sorted(file_db.keys())
    
    if len(file_db) > 0:
        if source_type == "auto":
            st.info(f"📁 Автоматически загружено: {len(file_db)} снимков (GitHub)")
        else:
            st.info(f"📂 Загружено вручную: {len(file_db)} снимков")
            
        st.markdown("---")
        start_date = st.selectbox("Дата старта", options=sorted_dates, format_func=lambda x: x.strftime("%d.%m.%Y"))
        horizon = st.slider("Горизонт (сут.)", 1, 7, 3)
        
        target_date = start_date + timedelta(days=horizon)
        has_truth = target_date in file_db
        
        st.write(f"Цель: **{target_date.strftime('%d.%m.%Y')}**")
        
        if has_truth:
            btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary")
        else:
            st.warning("⚠️ Нет файла для проверки")
            btn = False
    else:
        st.warning("Нет данных. Загрузите файлы на GitHub или сюда.")

# --- ЛОГИКА ---
if 'btn' in locals() and btn:
    try:
        status = st.status("Нейросетевое моделирование...", expanded=True)
        
        # Функция чтения (Универсальная: и для путей, и для загруженных файлов)
        def read_nc(f_item):
            if isinstance(f_item, str): # Если это путь к файлу на диске
                return xr.open_dataset(f_item)
            else: # Если это загруженный файл
                f_item.seek(0)
                # Читаем в память, так как xarray требует файл
                content = f_item.read()
                # Сохраняем временно, чтобы открыть
                temp_name = f"temp_{f_item.name}"
                with open(temp_name, "wb") as f: f.write(content)
                return xr.open_dataset(temp_name)

        # 1. СТАРТ
        ds = read_nc(file_db[start_date])
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        current_img = preprocess_raw_data(data_raw)
        
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ПРОГНОЗ
        prog_bar = status.progress(0)
        alpha = 0.75
        
        for day in range(1, horizon + 1):
            pred_ai = model.predict(input_batch, verbose=0)
            pred_stab = (input_batch * alpha) + (pred_ai * (1 - alpha))
            pred_clean = tf.where(pred_stab > 0.1, pred_stab, 0.0)
            input_batch = pred_clean
            
            status.write(f"✅ День {day}: Расчет завершен")
            prog_bar.progress(day / horizon)
        
        # Результат
        final_full = tf.image.resize(input_batch[0], [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status.update(label="Готово", state="complete", expanded=False)

        # 3. ФАКТ
        ds_t = read_nc(file_db[target_date])
        target_raw = ds_t[var_name].isel(time=0).squeeze().values
        target_clean = preprocess_raw_data(target_raw)
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # 4. МЕТРИКИ
        diff_map = np.abs(final_full - target_clean)
        diff_map[land_mask] = np.nan
        mae = np.nanmean(diff_map) * 100
        accuracy = 100 - mae

        # 5. ВИЗУАЛИЗАЦИЯ
        st.subheader(f"📊 Результат (Горизонт: {horizon} сут.)")
        col1, col2, col3 = st.columns(3)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E')
        
        with col1:
            st.markdown("### 🧠 Прогноз ИИ")
            fig1, ax1 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax1.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
        with col2:
            st.markdown("### 🛰️ Факт")
            fig2, ax2 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            ax2.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
        with col3:
            st.markdown("### 🔥 Ошибки")
            fig3, ax3 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
            cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white')
            ax3.axis('off')
            st.pyplot(fig3)
            
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность", f"{accuracy:.2f}%")
        m2.metric("MAE", f"{mae:.2f}%")
        m3.metric("Вердикт", "✅ УСПЕХ" if accuracy > 80 else "⚠️")

    except Exception as e:
        st.error(f"Ошибка: {e}")

elif len(file_db) == 0:
    st.info("В репозитории нет файлов fixed_*.nc. Загрузите их на GitHub или добавьте вручную.")

