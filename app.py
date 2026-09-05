import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (224, 224)

CUSTOM_MODEL_PATH = "models/best_custom_cnn.keras"
MOBILENET_MODEL_PATH = "models/mobilenetv2.keras"


# --------------------------------------------------
# Class names
# --------------------------------------------------

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# --------------------------------------------------
# Load both models
# --------------------------------------------------

@st.cache_resource
def load_models():
    custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

    return custom_model, mobilenet_model


custom_model, mobilenet_model = load_models()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def format_class_name(class_name):
    return class_name.replace("___", " → ").replace("_", " ")


def predict_image(model, image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    return [
        (class_names[i], float(predictions[i]))
        for i in top_indices
    ]


# --------------------------------------------------
# App
# --------------------------------------------------

st.title("🌿 Plant Disease Classifier")

st.markdown(
    """
Upload a plant leaf image and compare predictions from
**Custom CNN** and **MobileNetV2**.
"""
)


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Compare Models", use_container_width=True):

        with st.spinner("Analyzing with both models..."):

            custom_predictions = predict_image(
                custom_model,
                image
            )

            mobilenet_predictions = predict_image(
                mobilenet_model,
                image
            )

        # --------------------------------------------------
        # Side-by-side results
        # --------------------------------------------------

        st.subheader("Model Comparison")

        col1, col2 = st.columns(2)

        # --------------------------------------------------
        # Custom CNN
        # --------------------------------------------------

        with col1:

            st.markdown("## 🧠 Custom CNN")

            custom_class, custom_probability = custom_predictions[0]

            st.success(
                f"**{format_class_name(custom_class)}**"
            )

            st.metric(
                "Confidence",
                f"{custom_probability * 100:.2f}%"
            )

            st.markdown("### Top 3 Predictions")

            for rank, (class_name, probability) in enumerate(
                custom_predictions,
                start=1
            ):
                st.write(
                    f"**{rank}. {format_class_name(class_name)}**"
                )

                st.progress(
                    probability,
                    text=f"{probability * 100:.2f}%"
                )

        # --------------------------------------------------
        # MobileNetV2
        # --------------------------------------------------

        with col2:

            st.markdown("## 🚀 MobileNetV2")

            mobilenet_class, mobilenet_probability = mobilenet_predictions[0]

            st.success(
                f"**{format_class_name(mobilenet_class)}**"
            )

            st.metric(
                "Confidence",
                f"{mobilenet_probability * 100:.2f}%"
            )

            st.markdown("### Top 3 Predictions")

            for rank, (class_name, probability) in enumerate(
                mobilenet_predictions,
                start=1
            ):
                st.write(
                    f"**{rank}. {format_class_name(class_name)}**"
                )

                st.progress(
                    probability,
                    text=f"{probability * 100:.2f}%"
                )

        # --------------------------------------------------
        # Agreement
        # --------------------------------------------------

        st.divider()

        if custom_class == mobilenet_class:

            st.success(
                "✅ Both models predict the same disease."
            )

        else:

            st.warning(
                "⚠️ The models produced different predictions."
            )


# --------------------------------------------------
# Project information
# --------------------------------------------------

st.divider()

st.subheader("About the Project")

st.write(
    """
This project compares two CNN-based approaches for plant disease
classification using the PlantVillage dataset.

The models are:

- **Custom CNN** — lightweight model developed from scratch.
- **MobileNetV2** — pretrained on ImageNet and adapted using transfer learning.
"""
)

st.write(
    "**Dataset:** PlantVillage  |  "
    "**Classes:** 38  |  "
    "**Custom CNN Test Accuracy:** 93.40%  |  "
    "**MobileNetV2 Test Accuracy:** 96.18%"
)