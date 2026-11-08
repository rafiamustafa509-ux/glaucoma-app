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
def img_tag(path, width="100%"):
    """Inline base64 image (same approach as the preview, avoids the media endpoint)."""
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        return ""
    with open(full, "rb") as fh:
        enc = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{enc}" style="width:{width};border-radius:8px"/>'


def show_performance():
    st.subheader("Test-set performance (Model 3, cropped images)")
    st.caption("1,000 test images: 500 RG and 500 NRG. Threshold 0.5.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "87.0%")
    c2.metric("AUC", "0.9496")
    c3.metric("Loss", "0.3291")
    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", "89.7%")
    c5.metric("Recall (Sensitivity)", "83.6%")
    c6.metric("Specificity", "90.4%")

    st.markdown("#### Confusion matrix")
    st.markdown(f'<div style="text-align:center">{img_tag("perf_cm.png", "70%")}</div>',
                unsafe_allow_html=True)
    st.caption("TP = 418 glaucoma cases found, FN = 82 missed, "
               "TN = 452 healthy cleared, FP = 48 false alarms.")

    st.markdown("#### Threshold analysis (all 1,000 test images)")
    st.markdown(f'<div style="text-align:center">{img_tag("perf_threshold.png", "90%")}</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
| Threshold | Accuracy | Sensitivity | Specificity | Missed RG | False RG |
|---|---|---|---|---|---|
| 0.30 | 86.9% | 90.0% | 83.8% | 50 | 81 |
| 0.40 | 87.2% | 87.2% | 87.2% | 64 | 64 |
| **0.45 (app default)** | **87.3%** | **86.2%** | **88.4%** | 69 | 58 |
| 0.50 | 87.0% | 83.6% | 90.4% | 82 | 48 |
| 0.60 | 86.9% | 80.0% | 93.8% | 100 | 31 |
"""
    )
    st.caption("Lowering the threshold catches more glaucoma cases (higher sensitivity) "
               "but gives more false referrals. 0.45 gave the best balance and best accuracy.")

    st.markdown("#### Training curves (15 epochs, best checkpoint = epoch 13)")
    a, b = st.columns(2)
    a.markdown(img_tag("perf_loss.png"), unsafe_allow_html=True)
    b.markdown(img_tag("perf_acc.png"), unsafe_allow_html=True)

    st.markdown("#### Model comparison (12 experiments, accuracy / AUC)")
    st.markdown(
        """
| Model | Raw | Padded | Cropped |
|---|---|---|---|
| Model 1 (layer 193) | 0.883 / 0.9574 | 0.889 / 0.9467 | 0.874 / 0.9521 |
| Model 2 (layer 127) | 0.873 / 0.9455 | 0.877 / 0.9479 | 0.884 / 0.9593 |
| **Model 3 (layer 97)** | 0.860 / 0.9480 | 0.873 / 0.9445 | **0.889 / 0.9610** |
| Model 4 (layer 88) | 0.852 / 0.9445 | 0.848 / 0.9259 | 0.879 / 0.9526 |
"""
    )
    st.caption("Model 3 on cropped images gave the best AUC with only 111,521 parameters, "
               "so it was selected for deployment.")


def show_predictor():
    uploaded = st.file_uploader("Fundus image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
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

tab_predict, tab_perf = st.tabs(["🔍 Predict", "📊 Model Performance"])
with tab_predict:
    show_predictor()
with tab_perf:
    show_performance()

st.divider()
st.caption(
    "⚠️ This is a research/educational tool, not a medical diagnosis. "
    "Please consult an eye specialist for an actual diagnosis."
)
