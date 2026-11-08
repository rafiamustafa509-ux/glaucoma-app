import io
import base64
import os
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model3_crop.h5")
CLASS_PATH = os.path.join(BASE, "class_indices.json")
IMG_SIZE = 224

FULL_NAMES = {
    "RG": "Referable Glaucoma",
    "NRG": "Non-Referable Glaucoma",
}

st.set_page_config(page_title="Glaucoma Detection", page_icon="👁️", layout="centered")


def build_model(image_size=224):
    """Same architecture as the notebook (fallback if load_model fails)."""
    from tensorflow import keras
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import (
        Dropout, MaxPooling2D, Conv2D, GlobalMaxPooling2D, Dense,
    )

    inp = keras.Input(shape=(image_size, image_size, 3))
    base = keras.applications.MobileNetV3Large(
        weights=None, include_top=False, input_tensor=inp, alpha=0.75
    )
    x = base.get_layer(index=97).output
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 1, kernel_regularizer=regularizers.L1L2(l2=1e-3))(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 1, kernel_regularizer=regularizers.L1L2(l2=1e-3))(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inp, outputs=out)


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    from tensorflow import keras

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        model = build_model(IMG_SIZE)
        model.load_weights(MODEL_PATH)

    with open(CLASS_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def preprocess(img: Image.Image, center_crop: bool) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

    img = ImageOps.exif_transpose(img).convert("RGB")
    if center_crop:
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    # Training (flow_from_directory) used "nearest" interpolation by default
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)
    return preprocess_input(arr)


# ---------------- UI ----------------
st.title("👁️ Glaucoma Detection")
st.caption("MobileNetV3Large (Model 3, Crop data) • Upload a fundus image")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold (RG)", 0.05, 0.95, 0.45, 0.05)
    center_crop = st.checkbox(
        "Center-crop (square) before prediction",
        value=False,
        help="Enable only if the image is not already cropped. "
             "The model was trained on a cropped dataset.",
    )
    st.info(
        "The model was trained on **cropped fundus images**. "
        "For best results, upload a similarly cropped image."
    )

uploaded = st.file_uploader("Fundus image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(img_bytes))
    # Display image as an inline base64 data-URI. This bypasses Streamlit's
    # /media/ endpoint, which was returning a broken image on this machine.
    display_img = ImageOps.exif_transpose(image).convert("RGB")
    preview = display_img.copy()
    preview.thumbnail((700, 700))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<div style="text-align:center">'
        f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%;border-radius:8px"/>'
        f'<div style="font-size:0.85rem;opacity:0.7">Uploaded image</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("Predict", type="primary", use_container_width=True):
        model, idx_to_class = load_assets()
        with st.spinner("Predicting..."):
            x = preprocess(image, center_crop)
            prob_rg = float(model.predict(x, verbose=0)[0][0])  # sigmoid = P(class 1)

        label = idx_to_class[1] if prob_rg >= threshold else idx_to_class[0]
        conf = prob_rg if label == idx_to_class[1] else 1 - prob_rg

        if label == "RG":
            st.error(f"**{FULL_NAMES.get(label, label)}** ({label})")
        else:
            st.success(f"**{FULL_NAMES.get(label, label)}** ({label})")

        # Borderline warning: model is not sure when P(RG) is close to 0.5
        if 0.40 <= prob_rg <= 0.60:
            st.warning(
                "⚠️ Borderline case: P(RG) is between 0.40 and 0.60, so the model "
                "is not very sure about this image. Please do not rely on this "
                "result alone; a specialist should review it."
            )

        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{conf * 100:.1f}%")
        c2.metric("P(RG)", f"{prob_rg:.3f}")
        st.progress(min(max(prob_rg, 0.0), 1.0), text="Glaucoma (RG) probability")

st.divider()
st.caption(
    "⚠️ This is a research/educational tool, not a medical diagnosis. "
    "Please consult an eye specialist for an actual diagnosis."
)
