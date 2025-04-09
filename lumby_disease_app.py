import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the class labels
class_labels = {0: 'healthy', 1: 'lumpy'}

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_path="cow_lumpy_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define preprocessing function
def preprocess_image(image, target_size=(150, 150)):
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🐄 Cow Lumpy Skin Disease Detection")
st.write("Upload an image of a cow to check if it has lumpy skin disease.")

uploaded_file = st.file_uploader("Choose a cow image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output_data)
    confidence = float(np.max(output_data)) * 100

    st.write(f"🧠 Prediction: **{class_labels[prediction]}**")
    st.write(f"🔍 Confidence: **{confidence:.2f}%**")
