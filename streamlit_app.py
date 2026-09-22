# ============================================================
# streamlit_app.py
#
# Flower image classifier using Streamlit
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# 2. APPLICATION PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    APP_DIR
    /
    "flower_classifier.keras"
)

CLASS_NAMES_PATH = (
    APP_DIR
    /
    "class_names.json"
)


# ============================================================
# 3. IMAGE CONFIGURATION
# ============================================================

IMG_WIDTH = 160

IMG_HEIGHT = 160


# ============================================================
# 4. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Flower Image Classifier",

    page_icon="🌸",

    layout="centered"

)


# ============================================================
# 5. PAGE TITLE
# ============================================================

st.title(
    "🌸 Flower Image Classifier"
)


st.write(
    "Upload a flower image and the trained "
    "MobileNetV2 classifier will predict "
    "the flower class."
)


# ============================================================
# 6. LOAD CLASS NAMES
# ============================================================

def load_class_names():

    if CLASS_NAMES_PATH.exists():

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    return [

        "daisy",

        "dandelion",

        "roses",

        "sunflowers",

        "tulips"

    ]


CLASS_NAMES = load_class_names()


# ============================================================
# 7. CHECK MODEL FILE EXISTS
# ============================================================

if not MODEL_PATH.exists():

    st.error(

        "Model file not found.\n\n"
        "Expected file:\n\n"
        "flower_classifier.keras"

    )

    st.stop()


# ============================================================
# 8. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    loaded_model = tf.keras.models.load_model(

        MODEL_PATH,

        compile=False

    )

    return loaded_model


try:

    model = load_model()


except Exception as error:

    st.error(
        "The model exists but could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 9. CHECK MODEL OUTPUT
# ============================================================

if model.output_shape[-1] != len(CLASS_NAMES):

    st.error(

        f"Model has "
        f"{model.output_shape[-1]} outputs, "
        f"but {len(CLASS_NAMES)} "
        f"class names were provided."

    )

    st.stop()


# ============================================================
# 10. MODEL INFORMATION
# ============================================================

with st.expander(
    "Model information"
):

    st.write(
        "Architecture: MobileNetV2"
    )

    st.write(
        "Transfer learning: Feature Extraction"
    )

    st.write(
        f"Input shape: {model.input_shape}"
    )

    st.write(
        f"Output shape: {model.output_shape}"
    )

    st.write(
        f"Image size: "
        f"{IMG_WIDTH} × {IMG_HEIGHT}"
    )

    st.write(
        "Classes:"
    )

    st.write(
        CLASS_NAMES
    )


# ============================================================
# 11. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a flower image",

    type=[

        "jpg",

        "jpeg",

        "png"

    ]

)


# ============================================================
# 12. PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    image = Image.open(
        uploaded_file
    )


    image = image.convert(
        "RGB"
    )


    # ========================================================
    # SHOW IMAGE
    # ========================================================

    st.subheader(
        "Uploaded image"
    )


    st.image(

        image,

        caption="Uploaded image",

        use_container_width=True

    )


    # ========================================================
    # RESIZE IMAGE
    # ========================================================

    resized_image = image.resize(

        (

            IMG_WIDTH,

            IMG_HEIGHT

        )

    )


    # ========================================================
    # CONVERT TO NUMPY ARRAY
    # ========================================================

    image_array = np.array(

        resized_image,

        dtype=np.float32

    )


    # ========================================================
    # ADD BATCH DIMENSION
    # ========================================================
    #
    # Before:
    #
    # (160, 160, 3)
    #
    # After:
    #
    # (1, 160, 160, 3)
    #
    # ========================================================

    image_batch = np.expand_dims(

        image_array,

        axis=0

    )


    # ========================================================
    # DISPLAY PREPROCESSING INFORMATION
    # ========================================================

    with st.expander(
        "Preprocessing information"
    ):

        st.write(
            "Original image size:",
            image.size
        )

        st.write(
            "Resized image size:",
            resized_image.size
        )

        st.write(
            "Input tensor shape:",
            image_batch.shape
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================
    #
    # IMPORTANT:
    #
    # preprocess_input() is already included inside
    # the trained model.
    #
    # Therefore we DON'T call it here.
    #
    # ========================================================

    with st.spinner(
        "Running prediction..."
    ):

        predictions = model.predict(

            image_batch,

            verbose=0

        )[0]


    # ========================================================
    # GET BEST CLASS
    # ========================================================

    predicted_index = int(

        np.argmax(
            predictions
        )

    )


    predicted_class = CLASS_NAMES[

        predicted_index

    ]


    confidence = float(

        predictions[
            predicted_index
        ]

    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()


    st.subheader(
        "Prediction"
    )


    st.success(

        f"Predicted flower: "
        f"{predicted_class.upper()}"

    )


    st.metric(

        label="Confidence",

        value=(
            f"{confidence * 100:.2f}%"
        )

    )


    # ========================================================
    # CREATE PROBABILITY DATAFRAME
    # ========================================================

    probability_dataframe = pd.DataFrame(

        {

            "Flower":
                CLASS_NAMES,

            "Probability (%)":
                predictions * 100

        }

    )


    probability_dataframe = (
        probability_dataframe
        .sort_values(

            by="Probability (%)",

            ascending=False

        )
    )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.subheader(
        "Class probabilities"
    )


    st.dataframe(

        probability_dataframe,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # DISPLAY CHART
    # ========================================================

    st.subheader(
        "Probability chart"
    )


    chart_dataframe = (
        probability_dataframe
        .set_index(
            "Flower"
        )
    )


    st.bar_chart(
        chart_dataframe
    )


# ============================================================
# 13. NO IMAGE YET
# ============================================================

else:

    st.info(

        "Upload a JPG, JPEG or PNG image "
        "to start prediction."

    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()


st.caption(

    "Built with TensorFlow, "
    "MobileNetV2 and Streamlit."

)