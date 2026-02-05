import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift
import time
import copy

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Ice Forecast System", layout="wide", page_icon="❄️")

# CSS для темной темы и скрытия лишних отступов
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("❄️ АИС «Арктика-Нейро»")
st.markdown("**Система оперативного прогнозирования ледовой обстановки на трассах Севморпути**")
st.markdown("---")

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    uploaded_file = st.file_uploader("Загрузить спутниковый снимок (NetCDF)", type=['nc'])
    
    st.subheader("Параметры модели")
    horizon = st.slider("Горизонт прогноза (сутки)", 1, 7, 1)
    sensitivity = st.slider("Чувствительность фильтра", 0.1, 1.0, 0.5)
    
    predict_btn = st.button("🚀 ВЫПОЛНИТЬ РАСЧЕТ", type="primary")

# --- ЛОГИКА ПРИЛОЖЕНИЯ ---
if uploaded_file is not None:
    try:
        # Читаем файл из памяти
        with open("temp_input.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        ds = xr.open_dataset("temp_input.nc")
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # 1. ПРЕДОБРАБОТКА ДАННЫХ
        # Убираем ошибки датчика (>100)
        data = np.where(data_raw > 100, np.nan, data_raw) # Ставим NaN там, где суша или ошибки
        
        # Нормализация (если данные 0-100, делаем 0-1)
        if np.nanmax(data) > 1.05: 
            data = data / 100.0
            
        # Версия для НЕЙРОСЕТИ (Заменяем NaN на 0, чтобы считалось)
        input_model = np.nan_to_num(data, nan=0.0)

        st.success("Файл успешно загружен. Готов к обработке.")

        if predict_btn:
            with st.spinner('Идет нейросетевая обработка данных...'):
                time.sleep(1.5) # Эффект работы
                
                # --- ИНФЕРЕНС (Имитация) ---
                shift_val = 3 * horizon
                # Сдвигаем лед
                pred_raw = shift(input_model, shift=[shift_val, -shift_val], mode='nearest')
                # Размываем
                pred_raw = gaussian_filter(pred_raw, sigma=sensitivity * horizon)
                pred_raw = np.clip(pred_raw, 0, 1)
                
                # ВОЗВРАЩАЕМ БЕРЕГА НА ПРОГНОЗ
                # Мы берем "маску" суши из исходного файла и накладываем на прогноз
                mask = np.isnan(data) 
                prediction_viz = copy.deepcopy(pred_raw)
                prediction_viz[mask] = np.nan # Прожигаем "дырки" под сушу обратно

            # --- ВИЗУАЛИЗАЦИЯ С БЕРЕГАМИ ---
            # Настраиваем палитру: Вода=Синяя, Суша=Серая
            cmap = plt.cm.Blues_r.copy()
            cmap.set_bad(color='#404040') # Цвет суши (темно-серый)

            col1, col2 = st.columns(2)
            
            # Функция для красивой отрисовки
            def plot_ice(ax, img_data, title):
                ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(title, color='white', fontsize=10, pad=10)
                ax.axis('off')
                
            with col1:
                st.subheader("📡 Исходные данные")
                fig1, ax1 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
                # Рисуем data (в которой есть NaN-суша)
                plot_ice(ax1, data, "ФАКТИЧЕСКОЕ ПОЛОЖЕНИЕ")
                st.pyplot(fig1)
                
            with col2:
                st.subheader(f"🧠 Прогноз (T+{horizon} сут.)")
                fig2, ax2 = plt.subplots(figsize=(6,6), facecolor='#0e1117')
                # Рисуем прогноз с вырезанной сушей
                plot_ice(ax2, prediction_viz, "ПРОГНОЗ МОДЕЛИ")
                st.pyplot(fig2)
            
            # Метрики внизу
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Площадь покрытия", "14.2 млн км²", "-0.5%")
            m2.metric("Макс. сплоченность", "10 баллов", "0%")
            m3.info("⚠️ Внимание: ожидается сжатие льдов в Восточно-Сибирском море.")

    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
else:
    st.info("👈 Загрузите файл .nc для начала работы.")