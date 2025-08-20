import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Check if model exists
MODEL_PATH = "flowers_model.keras"
CLASS_NAMES_PATH = "class_names.txt"

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    st.error("⚠️ Model or class names file is missing!")
else:
    # Load model
    model = load_model(MODEL_PATH)

    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = f.read().splitlines()

    # Streamlit UI
    st.title("🌸 Flower Classification App")
    st.write("Upload a flower image and the model will predict its class using CNN.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
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
        except Exception as e:
            st.error(f"❌ Error processing the image: {e}")
