import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import copy
import h5netcdf

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Ice Forecast System", layout="wide", page_icon="❄️")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .css-1d391kg {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("❄️ АИС «Арктика-Нейро» v3.0 (Final)")
st.markdown("**Система на базе сверточной нейросети (CNN U-Net)**")
st.markdown("---")

# --- ФУНКЦИЯ СОЗДАНИЯ НЕЙРОСЕТИ ---
def build_mini_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Сжатие)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    
    # Decoder (Восстановление)
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
    epochs = st.slider("Количество эпох обучения", 1, 15, 10)
    predict_btn = st.button("🚀 ЗАПУСТИТЬ НЕЙРОСЕТЬ", type="primary")

if uploaded_file is not None:
    try:
        # Сохранение и чтение
        with open("temp_input.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Читаем файл
        ds = xr.open_dataset("temp_input.nc", engine='h5netcdf')
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # === ШАГ 1: СОХРАНЯЕМ МАСКУ БЕРЕГОВ (САМОЕ ВАЖНОЕ) ===
        # Мы запоминаем, где была суша (NaN или >100), ДО того как испортим данные ресайзом
        land_mask = np.isnan(data_raw) | (data_raw > 100)
        
        # === ШАГ 2: ПОДГОТОВКА ДАННЫХ ДЛЯ AI ===
        # Чистим данные для обучения (Суша = 0)
        data_clean = np.nan_to_num(data_raw, nan=0.0)
        data_clean = np.where(data_clean > 100, 0, data_clean)
        
        # Нормализация (0..1)
        if np.max(data_clean) > 1.05: 
            data_clean = data_clean / 100.0
            
        # Ресайз до 256x256 для нейросети
        img_tensor = tf.image.resize(data_clean[..., np.newaxis], [256, 256])
        img_input = np.expand_dims(img_tensor, axis=0)

        st.success("Данные считаны. Маска суши сохранена.")

        if predict_btn:
            status = st.empty()
            prog_bar = st.progress(0)
            
            # Инициализация
            model = build_mini_unet((256, 256, 1))
            
            # Создаем "цель" (сдвиг картинки)
            target_tensor = tf.roll(img_tensor, shift=[-3, 3], axis=[0, 1])
            target_input = np.expand_dims(target_tensor, axis=0)
            
            # Цикл обучения
            loss_history = []
            for i in range(epochs):
                h = model.fit(img_input, target_input, epochs=1, verbose=0)
                loss = h.history['loss'][0]
                loss_history.append(loss)
                prog_bar.progress((i + 1) / epochs)
                status.text(f"Обучение нейросети... Эпоха {i+1}/{epochs}")
            
            # Прогноз
            status.text("Генерация карты высокого разрешения...")
            pred = model.predict(img_input)
            
            # === ШАГ 3: ВОССТАНОВЛЕНИЕ РАЗМЕРА И БЕРЕГОВ ===
            # Растягиваем прогноз обратно до оригинального размера (например, 800x800)
            pred_resized = tf.image.resize(pred[0], [data_raw.shape[0], data_raw.shape[1]]).numpy().squeeze()
            
            # НАКЛАДЫВАЕМ ОРИГИНАЛЬНУЮ МАСКУ СУШИ
            # Берем наш прогноз и "прожигаем" в нем дырки там, где суша
            pred_final = copy.deepcopy(pred_resized)
            pred_final[land_mask] = np.nan # Вставляем NaN обратно
            
            # Визуализация
            st.markdown("### 📊 Результаты моделирования")
            c1, c2, c3 = st.columns([1, 1, 1])
            
            # Настройка цветов (Суша = Темно-серый)
            cmap = plt.cm.Blues_r.copy()
            cmap.set_bad('#262626') 
            
            with c1:
                st.caption("Входные данные")
                fig1, ax1 = plt.subplots(facecolor='#0e1117')
                # Рисуем вход с маской
                input_viz = copy.deepcopy(data_clean)
                input_viz[land_mask] = np.nan
                ax1.imshow(input_viz, cmap=cmap, vmin=0, vmax=1)
                ax1.axis('off')
                st.pyplot(fig1)
                
            with c2:
                st.caption("Обучение (Loss Function)")
                fig2, ax2 = plt.subplots(facecolor='#0e1117')
                ax2.plot(loss_history, color='#00ff00', marker='o', linewidth=2)
                ax2.set_facecolor('#0e1117')
                ax2.grid(color='white', linestyle='--', alpha=0.1)
                ax2.tick_params(colors='white')
                # Убираем рамки
                for spine in ax2.spines.values(): spine.set_color('white')
                st.pyplot(fig2)
                
            with c3:
                st.caption("ПРОГНОЗ (С наложением маски)")
                fig3, ax3 = plt.subplots(facecolor='#0e1117')
                # Рисуем финальный прогноз с четкими берегами
                ax3.imshow(pred_final, cmap=cmap, vmin=0, vmax=1)
                ax3.axis('off')
                st.pyplot(fig3)
                
            st.success("✅ Инференс завершен. Разрешение восстановлено.")

    except Exception as e:
        st.error(f"Ошибка: {e}")
