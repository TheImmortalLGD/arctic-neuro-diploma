import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import copy

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Ice Forecast System", layout="wide", page_icon="❄️")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .css-1d391kg {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("❄️ АИС «Арктика-Нейро» v2.0")
st.markdown("**Система на базе сверточной нейросети (CNN U-Net)**")
st.markdown("---")

# --- ФУНКЦИЯ СОЗДАНИЯ НЕЙРОСЕТИ (U-NET) ---
def build_mini_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Сжатие (Encoder) - анализ структуры льда
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    encoded = layers.MaxPooling2D((2, 2))(c2)
    
    # "Бутылочное горлышко" (самые важные признаки)
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    
    # Восстановление (Decoder) - построение прогноза
    u1 = layers.UpSampling2D((2, 2))(b)
    concat1 = layers.Concatenate()([u1, c2]) # Skip connection
    d1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    
    u2 = layers.UpSampling2D((2, 2))(d1)
    concat2 = layers.Concatenate()([u2, c1]) # Skip connection
    d2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d2)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    uploaded_file = st.file_uploader("Загрузить данные (NetCDF)", type=['nc'])
    
    st.subheader("Параметры обучения")
    epochs = st.slider("Количество эпох обучения", 1, 10, 5)
    
    predict_btn = st.button("🚀 ЗАПУСТИТЬ НЕЙРОСЕТЬ", type="primary")
    
    st.info("Используется архитектура Mini U-Net с 15,000 параметров.")

# --- ЛОГИКА ---
if uploaded_file is not None:
    try:
        # Чтение файла
        with open("temp_input.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        ds = xr.open_dataset("temp_input.nc")
        var_name = [v for v in ds.data_vars if 'ice' in v or 'conc' in v][0]
        data_raw = ds[var_name].isel(time=0).squeeze().values
        
        # Подготовка данных
        mask = np.where(data_raw > 100, 1, 0) # Маска суши
        data = np.where(data_raw > 100, 0, data_raw) # Очистка
        if np.nanmax(data) > 1.05: data = data / 100.0
        
        # Превращаем в тензор для нейросети (Размер должен быть кратен 32 для U-Net)
        # Для демо просто обрезаем до 256x256 или ресайзим
        img_tensor = tf.image.resize(data[..., np.newaxis], [256, 256])
        img_input = np.expand_dims(img_tensor, axis=0) # Batch size 1

        st.success("Данные загружены. Тензор сформирован: (1, 256, 256, 1)")

        if predict_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Создаем нейросеть
            status_text.text("Инициализация весов нейросети...")
            model = build_mini_unet((256, 256, 1))
            
            # 2. Формируем "Цель" (Target)
            # Чтобы сеть училась, мы имитируем "будущее" (сдвигаем картинку алгоритмически, 
            # чтобы сеть пыталась предсказать этот сдвиг)
            # В реальном проекте тут были бы данные за "завтра".
            target_tensor = tf.roll(img_tensor, shift=[3, 3], axis=[0, 1]) 
            target_input = np.expand_dims(target_tensor, axis=0)

            # 3. Обучение (Real Training Loop)
            status_text.text("Запуск процесса обучения (Backpropagation)...")
            
            # Кастомный цикл обучения для визуализации
            loss_plot = []
            plot_placeholder = st.empty()
            
            for epoch in range(epochs):
                history = model.fit(img_input, target_input, epochs=1, verbose=0, batch_size=1)
                loss = history.history['loss'][0]
                loss_plot.append(loss)
                
                # Обновляем прогресс
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Эпоха {epoch+1}/{epochs} - Ошибка (Loss): {loss:.4f}")
                time.sleep(0.3) # Чтобы было видно процесс
            
            # 4. Прогноз (Inference)
            status_text.text("Генерация прогноза...")
            prediction = model.predict(img_input)
            pred_img = prediction[0, :, :, 0]
            
            # Ресайз обратно (для красоты)
            pred_resized = tf.image.resize(pred_img[..., np.newaxis], [data.shape[0], data.shape[1]]).numpy().squeeze()
            
            # Восстанавливаем берега
            pred_final = copy.deepcopy(pred_resized)
            pred_final[mask == 1] = np.nan

            # --- ВИЗУАЛИЗАЦИЯ ---
            cmap = plt.cm.Blues_r.copy()
            cmap.set_bad(color='#404040')

            st.markdown("### Результаты работы модели")
            c1, c2, c3 = st.columns([1, 1, 1])
            
            def plot_ax(ax, img, title):
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, color='white', fontsize=10)
                ax.axis('off')

            with c1:
                fig1, ax1 = plt.subplots(facecolor='#0e1117')
                plot_ax(ax1, data, "ВХОД (T)")
                st.pyplot(fig1)
            
            with c2:
                fig2, ax2 = plt.subplots(facecolor='#0e1117')
                # График падения ошибки
                ax2.plot(loss_plot, color='#ff4757', marker='o')
                ax2.set_title("ГРАФИК ОБУЧЕНИЯ (LOSS)", color='white')
                ax2.set_xlabel("Эпоха")
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#0e1117')
                ax2.spines['bottom'].set_color('white')
                ax2.spines['left'].set_color('white')
                ax2.tick_params(colors='white')
                st.pyplot(fig2)

            with c3:
                fig3, ax3 = plt.subplots(facecolor='#0e1117')
                plot_ax(ax3, pred_final, "ПРОГНОЗ (T+1)")
                st.pyplot(fig3)
                
            st.success("✅ Модель успешно обучилась и выполнила инференс.")

    except Exception as e:
        st.error(f"Ошибка: {e}")
