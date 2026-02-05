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

# --- НАСТРОЙКИ ---
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

# --- МОДЕЛЬ ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Ошибка модели: {e}")
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

def clean_data_initial(d):
    """Очистка ТОЛЬКО для первого сырого кадра"""
    d = np.nan_to_num(d, nan=0.0)
    d = np.where(d > 100, 0, d)
    if np.max(d) > 1.05: d = d / 100.0 # Нормализация 0..1
    return d

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Данные")
    
    if model is None:
        st.error("❌ Файл модели не найден")
        st.stop()
    else:
        st.success("✅ Система готова")

    uploaded_files = st.file_uploader(
        "Загрузите файлы (Апрель)", 
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
        st.info(f"Снимков: {len(file_db)}")
        
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
        status = st.status("Диагностика и чтение данных...", expanded=True)
        
        # === ФУНКЦИЯ ЧТЕНИЯ С ВЫВОДОМ ОШИБОК ===
        def safe_read_debug(file_obj, temp_name):
            # 1. Запись
            file_obj.seek(0)
            with open(temp_name, "wb") as f:
                f.write(file_obj.read())
            
            # 2. Проверка размера (Сразу скажет, если это LFS ссылка)
            size = os.path.getsize(temp_name)
            if size < 3000:
                st.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл '{file_obj.name}' весит всего {size} байт!")
                st.warning("Это не спутниковый снимок, а текстовая ссылка GitHub LFS. Загрузите ОРИГИНАЛЬНЫЙ файл с вашего компьютера.")
                st.stop()

            # 3. Попытка открыть разными методами
            engines = ['netcdf4', 'h5netcdf', 'scipy']
            errors_log = []
            
            for eng in engines:
                try:
                    ds = xr.open_dataset(temp_name, engine=eng)
                    # Если открылось - ура, возвращаем
                    return ds
                except Exception as e:
                    errors_log.append(f"Движок '{eng}': {str(e)}")
            
            # Если дошли сюда - значит ни один не открыл
            st.error(f"❌ Не удалось открыть файл '{file_obj.name}'. Отчет об ошибках:")
            for err in errors_log:
                st.code(err, language='text')
            
            st.info("💡 ПОДСКАЗКА: Если ошибка 'HDF error' - файл битый. Если 'libnetcdf not found' - нет packages.txt.")
            st.stop()

        # 1. ЧТЕНИЕ СТАРТА
        ds = safe_read_debug(file_db[start_date], "temp_start.nc")
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Маска
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        
        # Первичная очистка
        current_img = clean_data_initial(data_raw)
        
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
        
        # Результат
        final_full = tf.image.resize(input_batch[0], [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status.update(label="Готово", state="complete", expanded=False)

        # 3. ФАКТ
        ds_t = safe_read_debug(file_db[target_date], "temp_target.nc")
        target_raw = ds_t[var_name].isel(time=0).squeeze().values
        target_clean = clean_data_initial(target_raw)
        
        target_viz = copy.deepcopy(target_clean)
        target_viz[land_mask] = np.nan
        
        # 4. МЕТРИКИ
        diff_map = np.abs(final_full - target_clean)
        diff_map[land_mask] = np.nan
        
        mae = np.nanmean(diff_map) * 100
        accuracy = 100 - mae

        # 5. ВИЗУАЛИЗАЦИЯ
        st.subheader(f"📊 Результат (Горизонт: {horizon} сут.)")
        
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
            st.markdown("### 🔥 Ошибки")
            fig3, ax3 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
            ax3.axis('off')
            st.pyplot(fig3)
            
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность", f"{accuracy:.2f}%")
        m2.metric("MAE", f"{mae:.2f}%")
        m3.metric("Статус", "✅ НОРМА" if accuracy > 80 else "⚠️")

    except Exception as e:
        st.error(f"Системная ошибка: {e}")

elif not uploaded_files:
    st.info("👈 Загрузите файлы.")
