from fastai.learner import load_learner
import streamlit as st
import fastai.vision
from PIL import Image
from fastai.vision.core import PILImage


def load_image(image):

    return Image.open(image)


def predict_img(img):

    if img is not None:
        return learner_inf.predict(pil_img)


learner_inf = load_learner("./export.pkl")

pic = st.file_uploader("Upload Files")

probs = []
pred_idx = 1
pred = "n/a"

# Display image
if pic is not None:
    img = load_image(pic)
    st.image(img)

    # Parse image
    pil_img = PILImage.create(pic)

    # Predict category
    pred, pred_idx, probs = predict_img(pil_img)

# Classify
if st.button("Classify"):
    if str(pred) in ("no_mask", "beard"):
        pred = "No mask"
    else:
        pred = "Mask"
    "Prediction: ", pred
    "Probability: ", str(round(probs[pred_idx].item(), 5))
