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
st.set_page_config(page_title="Arctic-PRO: Validation Suite", layout="wide", page_icon="🧊")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .stMetric {background-color: #1e212b; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧊 АИС «Арктика-PRO»: Валидация на временных рядах")
st.markdown("**Эксперимент: Проверка точности модели на данных АПРЕЛЯ (Unseen Data)**")
st.info("ℹ️ Режим работы: Модель обучена на МАРТЕ. Тестирование проводится на АПРЕЛЕ.")
st.markdown("---")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_ai_model():
    # Ищем файл модели
    if not os.path.exists('ice_model_month_v2.h5'): return None
    return load_model('ice_model_month_v2.h5')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Ошибка модели: {e}")
    model = None

# --- ФУНКЦИЯ ПОИСКА ДАТЫ В ИМЕНИ ФАЙЛА ---
def extract_date(filename):
    # Ищет паттерн YYYYMMDD (например, 20200401)
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except:
            return None
    return None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("🗂️ Загрузка архива")
    
    if model is None:
        st.error("❌ Файл ice_model_month_v2.h5 не найден!")
        st.stop()
    else:
        st.success("✅ Нейросеть (Март) активна")
    
    # МУЛЬТИ-ЗАГРУЗКА
    uploaded_files = st.file_uploader(
        "Шаг 1. Загрузите файлы (31 марта + Апрель)", 
        type=['nc'], 
        accept_multiple_files=True
    )
    
    # Индексация файлов
    file_db = {} # Словарь: {Дата : Файл}
    if uploaded_files:
        for f in uploaded_files:
            dt = extract_date(f.name)
            if dt:
                file_db[dt] = f
        
        # Сортируем даты
        sorted_dates = sorted(file_db.keys())
        st.success(f"Распознано снимков: {len(file_db)}")
        
        if len(file_db) > 0:
            st.markdown("---")
            st.header("⚙️ Параметры эксперимента")
            
            # ВЫБОР ДАТЫ СТАРТА
            start_date = st.selectbox(
                "Шаг 2. Дата старта (Начальные условия)", 
                options=sorted_dates,
                format_func=lambda x: x.strftime("%d.%m.%Y")
            )
            
            # ВЫБОР ГОРИЗОНТА
            # Ограничиваем горизонт, чтобы не выйти за пределы загруженных файлов
            max_horizon = 14
            horizon = st.slider("Шаг 3. Горизонт прогноза (суток)", 1, max_horizon, 3)
            
            # ВЫЧИСЛЕНИЕ ЦЕЛЕВОЙ ДАТЫ
            target_date = start_date + timedelta(days=horizon)
            has_truth = target_date in file_db
            
            st.markdown(f"**Целевая дата:** `{target_date.strftime('%d.%m.%Y')}`")
            
            if has_truth:
                st.info("✅ Файл для проверки найден в загрузках")
                btn_disabled = False
            else:
                st.warning("⚠️ Файла за эту дату нет. Сравнение невозможно.")
                btn_disabled = True
            
            predict_btn = st.button("🚀 ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ", type="primary", disabled=btn_disabled)

# --- ОСНОВНАЯ ЛОГИКА ---
if 'predict_btn' in locals() and predict_btn:
    try:
        # Контейнер для статуса
        status_container = st.status("Инициализация вычислительного ядра...", expanded=True)
        
        # 1. ЧТЕНИЕ СТАРТОВОГО ФАЙЛА
        start_file_obj = file_db[start_date]
        start_file_obj.seek(0) # Сброс курсора
        
        with open("start_temp.nc", "wb") as f: f.write(start_file_obj.read())
        
        ds = xr.open_dataset("start_temp.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Подготовка данных
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        orig_shape = data_raw.shape
        
        def clean(d):
            d = np.nan_to_num(d, nan=0.0)
            d = np.where(d > 100, 0, d)
            if np.max(d) > 1.05: d = d / 100.0
            return d

        current_img = clean(data_raw)
        
        # Тензор для входа
        input_tensor = tf.image.resize(current_img[..., np.newaxis], [256, 256])
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # 2. ЦИКЛ ПРОГНОЗИРОВАНИЯ (РЕКУРСИЯ)
        progress_bar = status_container.progress(0)
        
        for day in range(1, horizon + 1):
            # Инференс
            pred = model.predict(input_batch, verbose=0)
            
            # Выход становится входом для следующего дня
            input_batch = pred 
            
            # Обновление статуса
            sim_date = start_date + timedelta(days=day)
            status_container.write(f"✅ День {day} ({sim_date.strftime('%d.%m')}): Расчет дрейфа завершен")
            progress_bar.progress(day / horizon)
            time.sleep(0.2) # Имитация нагрузки для наглядности
            
        # 3. ПОСТ-ОБРАБОТКА РЕЗУЛЬТАТА
        final_small = input_batch[0]
        final_full = tf.image.resize(final_small, [orig_shape[0], orig_shape[1]]).numpy().squeeze()
        
        # Восстанавливаем маску суши
        final_viz = copy.deepcopy(final_full)
        final_viz[land_mask] = np.nan
        
        status_container.update(label="Расчет завершен успешно!", state="complete", expanded=False)

        # 4. СРАВНЕНИЕ С ФАКТОМ
        target_file_obj = file_db[target_date]
        target_file_obj.seek(0)
        with open("target_temp.nc", "wb") as f: f.write(target_file_obj.read())
        
        ds_target = xr.open_dataset("target_temp.nc", engine='h5netcdf')
        target_raw = ds_target[var_name].isel(time=0).squeeze().values
        target_clean = clean(target_raw)
        
        # Расчет ошибки (MAE)
        diff = np.abs(final_full - target_clean)
        diff[land_mask] = np.nan # Игнорируем сушу
        mae = np.nanmean(diff) * 100 # В процентах
        accuracy = 100 - mae
        
        # 5. ВИЗУАЛИЗАЦИЯ
        st.subheader(f"📊 Отчет о валидации ({start_date.strftime('%d.%m')} ➝ {target_date.strftime('%d.%m')})")
        
        col1, col2, col3 = st.columns(3)
        cmap = plt.cm.Blues_r.copy()
        cmap.set_bad('#1E1E1E') # Цвет суши
        
        with col1:
            st.caption("1. СТАРТ (Исходные данные)")
            fig1, ax1 = plt.subplots(facecolor='#0e1117')
            start_viz = copy.deepcopy(current_img)
            start_viz[land_mask] = np.nan
            ax1.imshow(start_viz, cmap=cmap, vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.caption(f"2. ПРОГНОЗ НЕЙРОСЕТИ (+{horizon} сут.)")
            fig2, ax2 = plt.subplots(facecolor='#0e1117')
            ax2.imshow(final_viz, cmap=cmap, vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2)
            
        with col3:
            st.caption("3. ФАКТ (Спутниковый контроль)")
            fig3, ax3 = plt.subplots(facecolor='#0e1117')
            target_viz = copy.deepcopy(target_clean)
            target_viz[land_mask] = np.nan
            ax3.imshow(target_viz, cmap=cmap, vmin=0, vmax=1)
            ax3.axis('off')
            st.pyplot(fig3)
        
        # МЕТРИКИ
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Точность прогноза", f"{accuracy:.2f}%", help="100% - средняя ошибка")
        m2.metric("Горизонт планирования", f"{horizon} суток")
        m3.metric("Статус теста", "УСПЕХ" if accuracy > 80 else "ТРЕБУЕТ КАЛИБРОВКИ", 
                 delta="Pass" if accuracy > 80 else "-Fail")
        
        # КАРТА ОШИБОК
        with st.expander("🔎 Детальный анализ ошибок (Тепловая карта)"):
            fig_err, ax_err = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            diff_viz = copy.deepcopy(diff)
            im = ax_err.imshow(diff_viz, cmap='hot', vmin=0, vmax=0.4) # Ошибки > 40% ярко-белые
            plt.colorbar(im, ax=ax_err, label="Величина отклонения")
            ax_err.set_title("Зоны расхождения прогноза с фактом", color='white')
            ax_err.axis('off')
            st.pyplot(fig_err)

    except Exception as e:
        st.error(f"Критическая ошибка: {e}")

elif not uploaded_files:
    st.info("👋 Привет! Чтобы начать, выделите все файлы .nc за апрель и перетащите их в панель слева.")
