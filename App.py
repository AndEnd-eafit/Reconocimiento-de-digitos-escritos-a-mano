import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@700&family=Inter:wght@400&display=swap');
        body {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Lexend', sans-serif;
        }
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Función para predecir el dígito
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
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Imagen centrada
image = Image.open('tu_imagen.png')
st.image(image, width=200)

# Parámetros del canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # Color del trazo
bg_color = '#000000'  # Color de fondo

# Crear el componente de canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno con opacidad
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Botón para predecir
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('El Dígito es: ' + str(res))
    else:
        st.header('Por favor dibuja en el canvas el dígito.')

# Barra lateral
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalúa ")
st.sidebar.text("la capacidad de una RNA para reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en el desarrollo de Vinay Uniyal")

