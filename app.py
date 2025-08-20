import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("flowers_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = f.read().splitlines()

# Streamlit UI
st.title("🌸 Flower Classification App")
st.write("Upload a flower image and the model will predict its class using CNN.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"✅ Predicted class: **{predicted_class}** with confidence {confidence:.2f}%")

