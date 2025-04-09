import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("cow_lumpy_disease_model.keras")

# Define class labels
class_labels = {0: 'healthy', 1: 'lumpy'}

# Define image size
img_size = (150, 150)

st.title("Cow Lumpy Disease Detection")
st.write("Upload a cow image to predict whether it has lumpy disease or not.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class.upper()}**")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
