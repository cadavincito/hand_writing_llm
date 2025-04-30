import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Función de predicción
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Configuración de la página
st.set_page_config(page_title='Reconocimiento de Dígitos', layout="wide", page_icon="✏️")

# CSS personalizado - Light Mode
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Tema claro */
        body {
            background-color: #f5f5f5;
            color: #333333;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 1rem auto;
            max-width: 1200px;
            text-align: center;
        }
        
        /* Encabezados */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2d3748 !important;
            text-align: center;
        }
        
        /* Canvas container */
        .canvas-container {
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem;
            background-color: #f8fafc;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin: 1.5rem auto;
            display: flex;
            justify-content: center;
            max-width: 450px;
        }
        
        /* Botones */
        .stButton button {
            background: linear-gradient(135deg, #4299e1, #3182ce);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 1rem auto;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(66, 153, 225, 0.3);
            background: linear-gradient(135deg, #3182ce, #4299e1);
        }
        
        /* Barra lateral */
        .stSidebar {
            background: linear-gradient(180deg, #ffffff, #f7fafc);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #2d3748 !important;
            text-align: center;
        }
        .stSidebar p, .stSidebar .stText {
            color: #4a5568 !important;
            text-align: center;
        }
        
        /* Slider */
        .stSlider {
            margin: 1.5rem auto;
            width: 80%;
        }
        .stSlider .st-ae {
            color: #3182ce !important;
        }
        .stSlider .st-af {
            background-color: #3182ce !important;
        }
        
        /* Resultados */
        .stAlert {
            border-radius: 12px !important;
            margin: 1rem auto !important;
            max-width: 500px;
        }
        
        /* Texto del slider */
        .stSlider label {
            color: #4a5568 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo
st.markdown("<h1 style='color: #2d3748;'>✏️ Reconocimiento de Dígitos</h1>", unsafe_allow_html=True)
st.subheader("Dibuja un dígito (0-9) en el panel y presiona 'Predecir'")

# Canvas centrado
stroke_width = st.slider('Ancho del pincel', 1, 30, 15, help="Ajusta el grosor del trazo al dibujar")

st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(66, 153, 225, 0.2)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#ffffff",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)
st.markdown('</div>', unsafe_allow_html=True)

# Botón de predicción
if st.button('🔍 Predecir Dígito', use_container_width=True):
    if canvas_result.image_data is not None:
        with st.spinner("Analizando..."):
            try:
                input_numpy_array = np.array(canvas_result.image_data)
                input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
                res = predictDigit(input_image)
                st.success(f"**El dígito reconocido es:** {res}")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {str(e)}")
    else:
        st.warning("Por favor dibuja un dígito en el canvas primero")

# Barra lateral
with st.sidebar:
    st.markdown("<h2 style='color: #2d3748;'>Sobre esta App</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color: #4a5568;'>
        Esta aplicación utiliza una red neuronal convolucional (CNN) entrenada
        para reconocer dígitos escritos a mano (0-9).
        </p>
        <p style='color: #4a5568;'>
        Basado en el trabajo de Vinay Uniyal
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 style='color: #2d3748;'>Instrucciones</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ol style='color: #4a5568; text-align: left;'>
            <li>Ajusta el ancho del pincel</li>
            <li>Dibuja un dígito en el área central</li>
            <li>Presiona el botón "Predecir"</li>
        </ol>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.caption("Aplicación desarrollada con Streamlit y TensorFlow | © 2023 Reconocimiento de Dígitos")
