import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")

st.title("🩺 Skin Lesion CNN Classifier")
st.write("Upload a skin lesion image to classify it across 7 categories.")
st.warning("⚠️ **Medical Disclaimer:** This AI tool is for educational purposes only and is not a medical diagnosis. Please consult a dermatologist.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.h5")

model = load_model()

CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]

uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_batch)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index] * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

    st.subheader("Confidence Breakdown")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name}:** {predictions[i]*100:.2f}%")
        st.progress(float(predictions[i]))
