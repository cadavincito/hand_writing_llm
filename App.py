import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

# Estilos CSS
st.markdown("""
    <style>
    .stApp {
        text-align: center;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        text-align: center;
    }
    .stSlider {
        width: 50%;
        margin: auto;
    }
    .stButton {
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("✏️ Dibuja el dígito en el panel y presiona 'Predecir'")

st.write("")  # Espacio

# Parámetros del canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Organizar el canvas centrado usando columnas
col1, col2, col3 = st.columns([1,2,1])  # Hacemos la del medio más grande
with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=400,   # Canvas más grande
        width=400,
        key="canvas",
    )

st.write("")  # Espacio

# Botón de predicción
if st.button('🔮 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('✏️ El dígito es: ' + str(res))
    else:
        st.header('⚠️ Por favor dibuja en el canvas el dígito.')

# Sidebar
st.sidebar.title("Acerca de:")
st.sidebar.text("Esta aplicación evalúa ")
st.sidebar.text("la capacidad de una RNA de reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en el desarrollo de Vinay Uniyal")
