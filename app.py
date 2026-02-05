import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import copy
import h5netcdf # Явный импорт для надежности

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Ice Forecast System", layout="wide", page_icon="❄️")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .css-1d391kg {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("❄️ АИС «Арктика-Нейро» v2.1")
st.markdown("**Система на базе сверточной нейросети (CNN U-Net)**")
st.markdown("---")

# --- ФУНКЦИЯ СОЗДАНИЯ НЕЙРОСЕТИ ---
def build_mini_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    
    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# --- ИНТЕРФЕЙС ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    uploaded_file = st.file_uploader("Загрузить данные (NetCDF)", type=['nc'])
    epochs = st.slider("Количество эпох обучения", 1, 15, 5)
    predict_btn = st.button("🚀 ЗАПУСТИТЬ НЕЙРОСЕТЬ", type="primary")

if uploaded_file is not None:
    try:
        # Сохранение и чтение
        with open("temp_input.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Чтение с безопасным движком
        ds = xr.open_dataset("temp_input.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # === ГЛАВНОЕ ИСПРАВЛЕНИЕ ===
        # 1. Сначала убираем технические NaN из файла
        data_raw = np.nan_to_num(data_raw, nan=0.0)
        
        # 2. Создаем маску суши (все что > 100 или было 0, если это маска)
        # Обычно 255 или >100 - это маска
        mask = np.where(data_raw > 100, 1, 0)
        
        # 3. Чистим данные для нейросети (Суша = 0.0)
        data = np.where(data_raw > 100, 0, data_raw)
        
        # 4. Нормализация (0..1)
        if np.max(data) > 1.05: 
            data = data / 100.0
            
        # Еще раз страхуемся от мусора
        data = np.nan_to_num(data, nan=0.0)
        # ===========================

        # Подготовка тензора (256x256)
        img_tensor = tf.image.resize(data[..., np.newaxis], [256, 256])
        img_input = np.expand_dims(img_tensor, axis=0)

        st.success("Данные прошли валидацию и очистку.")

        if predict_btn:
            status = st.empty()
            prog_bar = st.progress(0)
            
            # Инициализация
            model = build_mini_unet((256, 256, 1))
            
            # Создаем "цель" (сдвиг картинки, имитация дрейфа)
            target_tensor = tf.roll(img_tensor, shift=[-5, 5], axis=[0, 1])
            target_input = np.expand_dims(target_tensor, axis=0)
            
            # Цикл обучения
            loss_history = []
            for i in range(epochs):
                h = model.fit(img_input, target_input, epochs=1, verbose=0)
                loss = h.history['loss'][0]
                loss_history.append(loss)
                
                prog_bar.progress((i + 1) / epochs)
                status.text(f"Обучение: Эпоха {i+1}/{epochs} | Ошибка: {loss:.4f}")
            
            # Прогноз
            status.text("Генерация карты...")
            pred = model.predict(img_input)
            
            # Обработка результата
            pred_img = tf.image.resize(pred[0], [data.shape[0], data.shape[1]]).numpy().squeeze()
            
            # Накладываем сушу обратно (для красоты)
            pred_viz = copy.deepcopy(pred_img)
            pred_viz[mask == 1] = np.nan 
            
            # Визуализация
            st.markdown("### Результаты моделирования")
            c1, c2, c3 = st.columns([1, 1, 1])
            
            cmap = plt.cm.Blues_r.copy()
            cmap.set_bad('#404040') # Цвет суши
            
            with c1:
                st.caption("Входные данные")
                fig1, ax1 = plt.subplots(facecolor='#0e1117')
                # Для входа тоже используем маску, чтобы было красиво
                data_viz = copy.deepcopy(data)
                data_viz[mask == 1] = np.nan
                ax1.imshow(data_viz, cmap=cmap, vmin=0, vmax=1)
                ax1.axis('off')
                st.pyplot(fig1)
                
            with c2:
                st.caption("Динамика обучения")
                fig2, ax2 = plt.subplots(facecolor='#0e1117')
                ax2.plot(loss_history, color='#00ff00', marker='o')
                ax2.set_facecolor('#0e1117')
                ax2.grid(color='gray', linestyle='--', alpha=0.3)
                ax2.tick_params(colors='white')
                # Убираем рамки
                for spine in ax2.spines.values(): spine.set_edgecolor('white')
                st.pyplot(fig2)
                
            with c3:
                st.caption("Прогноз (CNN Output)")
                fig3, ax3 = plt.subplots(facecolor='#0e1117')
                ax3.imshow(pred_viz, cmap=cmap, vmin=0, vmax=1)
                ax3.axis('off')
                st.pyplot(fig3)
                
            st.success("✅ Инференс успешно завершен")

    except Exception as e:
        st.error(f"Ошибка обработки: {e}")
