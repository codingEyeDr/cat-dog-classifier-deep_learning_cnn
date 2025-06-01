import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 100
class_name = ['cat', 'dog']

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_dog_model.keras')
    return model

model = load_model()

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Image Classifier")
st.write("Upload an image, and the model will predict whether it's a **cat** or a **dog**.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image using PIL
    img = image.resize((IMG_SIZE, IMG_SIZE))  # Resize
    img = np.array(img)  # Convert to numpy array
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and add batch dimension

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # Show result
    st.markdown(f"### 🎯 Prediction: **{class_name[class_index]}** ({confidence:.2f}% confidence)")
