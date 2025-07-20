import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model
model_path = 'brain_tumor_classifier.keras1'
model = load_model(model_path)

# Tumor types
tumor_types = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# Streamlit app
st.title("Brain Tumor Classifier")
st.write("Upload an MRI image of a brain to get the classification.")

# Create a directory for temporary files if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Save the uploaded file to a temporary location
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Load the image and preprocess it
    img_path = os.path.join("temp", uploaded_file.name)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    tumor_type = tumor_types[predicted_class]
    st.write(f"Prediction: {tumor_type}")
