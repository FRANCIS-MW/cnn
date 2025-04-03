import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.pkl")

# Streamlit UI
def main():
    st.title("MNIST Digit Classifier")
    st.write("Draw a digit below and the model will predict it!")
    
    # Canvas for drawing
    canvas = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])
    
    if canvas is not None:
        image = Image.open(canvas).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image)  # Convert to numpy array
        image = cv2.bitwise_not(image)  # Invert colors (black background, white digit)
        image = image / 255.0  # Normalize
        image = image.reshape(1, 28, 28, 1)  # Reshape for model input
        
        # Model prediction
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        
        st.image(image.reshape(28, 28), caption="Processed Image", width=150)
        st.write(f"### Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
