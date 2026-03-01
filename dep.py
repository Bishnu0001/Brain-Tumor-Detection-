import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load Model
model = tf.keras.models.load_model("C:\\Spyder\\NLP\\project_img\\brain_tumor_model.h5")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor Detection App")
st.write("Upload an MRI Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess Image
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write("### Prediction:", class_names[pred_class])
    st.write("Confidence:", round(float(confidence)*100,2), "%")
