import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "flower_classifier.h5"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"

IMG_WIDTH = 160
IMG_HEIGHT = 160


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Flower Image Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Flower Image Classifier")

st.write(
    "Upload a flower image and the trained MobileNetV2 classifier "
    "will predict the flower class."
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    # Fallback for the TensorFlow Flowers dataset
    return [
        "daisy",
        "dandelion",
        "roses",
        "sunflowers",
        "tulips",
    ]


CLASS_NAMES = load_class_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


model = load_model()


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:
    st.error(
        "Model file not found. Add flower_classifier.h5 to the same "
        "directory as streamlit_app.py, then restart the app."
    )
    st.stop()

if model.output_shape[-1] != len(CLASS_NAMES):
    st.error(
        f"Model output has {model.output_shape[-1]} classes, "
        f"but class_names.json contains {len(CLASS_NAMES)} classes."
    )
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model information"):
    st.write("Architecture: MobileNetV2 transfer learning")
    st.write(f"Input shape: {model.input_shape}")
    st.write(f"Output shape: {model.output_shape}")
    st.write(f"Image size: {IMG_WIDTH} × {IMG_HEIGHT}")
    st.write(f"Classes: {', '.join(CLASS_NAMES)}")


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------
    #
    # The saved training model already contains
    # MobileNetV2 preprocess_input() inside the model graph.
    #
    # Therefore the Streamlit app only needs to:
    #   1. convert to RGB
    #   2. resize to 160 x 160
    #   3. convert to float32
    #   4. add the batch dimension
    #
    # --------------------------------------------------------

    resized_image = image.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32,
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0,
    )

    with st.spinner("Running prediction..."):
        predictions = model.predict(
            image_batch,
            verbose=0,
        )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Prediction")

    st.success(
        f"Predicted class: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    # --------------------------------------------------------
    # PROBABILITY TABLE
    # --------------------------------------------------------

    probability_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Probability (%)": predictions * 100,
        }
    ).sort_values(
        "Probability (%)",
        ascending=False,
    )

    st.subheader("Class probabilities")

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    chart_df = probability_df.set_index("Class")

    st.subheader("Probability chart")

    st.bar_chart(
        chart_df
    )

else:
    st.info(
        "Upload a JPG, JPEG, or PNG image to start."
    )


st.divider()

st.caption(
    "Built with TensorFlow, MobileNetV2 and Streamlit."
)
