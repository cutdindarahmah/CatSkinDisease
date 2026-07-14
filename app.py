import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ===========================================
# Konfigurasi Halaman
# ===========================================
st.set_page_config(
    page_title="Cat Skin Disease Detection",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Cat Skin Disease Detection")
st.write("Deteksi penyakit kulit pada kucing menggunakan Deep Learning.")

MODEL_DIR = Path("models")
EFFICIENTNET_PATH = MODEL_DIR / "efficientnetb1_cat_skin_disease_final.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenetv2_cat_skin_disease.h5"

CLASS_NAMES = [
    "Flea_Allergy",
    "Health",
    "Ringworm",
    "Scabies"
]

@st.cache_resource
def load_models():
    models = {}

    if EFFICIENTNET_PATH.exists():
        models["EfficientNetB1"] = tf.keras.models.load_model(str(EFFICIENTNET_PATH))

    if MOBILENET_PATH.exists():
        models["MobileNetV2"] = tf.keras.models.load_model(str(MOBILENET_PATH))

    return models

models = load_models()

if not models:
    st.error("Model tidak ditemukan. Pastikan file model ada di folder `models/`.")
    st.stop()

model_option = st.selectbox("Pilih Model", list(models.keys()))
model = models[model_option]

uploaded_file = st.file_uploader(
    "Upload Gambar Kucing",
    type=["jpg", "jpeg", "png"]
)


def get_input_size(model):
    input_shape = model.input_shape
    if hasattr(input_shape, '__len__') and len(input_shape) >= 4:
        _, height, width, _ = input_shape
        if height is not None and width is not None:
            return int(width), int(height)
    return 224, 224


def preprocess_image(image: Image.Image, size, model_name):
    image = image.resize(size)
    image_array = np.array(image).astype("float32")
    if model_name == "EfficientNetB1":
        image_array = efficientnet_preprocess(image_array)
    elif model_name == "MobileNetV2":
        image_array = mobilenet_preprocess(image_array)
    else:
        image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)


if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka file: {e}")
    else:
        st.image(
            image,
            caption="Gambar yang Dipilih",
            use_container_width=True
        )

        if st.button("Prediksi Sekarang"):
            with st.spinner("Memproses gambar..."):
                try:
                    input_size = get_input_size(model)
                    img_array = preprocess_image(image, input_size, model_option)
                    prediction = model.predict(img_array, verbose=0)
                    predicted_index = int(np.argmax(prediction[0]))
                    confidence = float(prediction[0][predicted_index] * 100)

                    st.success(f"Hasil Prediksi: {CLASS_NAMES[predicted_index]}")
                    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")

                    st.subheader("Probabilitas Semua Kelas")
                    for label, score in zip(CLASS_NAMES, prediction[0]):
                        st.write(f"**{label}** : {score * 100:.2f}%")
                        st.progress(float(score))
                except Exception as e:
                    st.error(f"Gagal memprediksi gambar: {e}")
        else:
            st.info("Tekan tombol Prediksi Sekarang untuk melihat hasil.")
else:
    st.info("Silakan upload gambar kucing untuk melihat hasil prediksi.")