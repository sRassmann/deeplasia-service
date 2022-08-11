"""
Based on https://github.com/Joshmantova/Eagle-Vision/
"""
import cv2
from PIL import Image
import os
import torch

import streamlit as st
import pandas as pd
import numpy as np

from fscnn.lib.models import MaskModel
from fscnn.predict import Predictor as MaskPredictor
from bone_age.models import Predictor as AgePredictor


@st.cache()
def load_fscnn(path: str = "./models/fscnn_cos.ckpt") -> MaskModel:
    return MaskPredictor(checkpoint=path)


@st.cache()
def load_age_model(use_cuda=False) -> MaskModel:
    return AgePredictor(use_cuda=use_cuda)


if __name__ == "__main__":
    mask_predictor = load_fscnn()
    age_predictor = load_age_model()

    if torch.cuda.is_available():
        if st.checkbox("use GPU"):
            st.write("Inference on GPU")
            age_predictor = load_age_model(use_cuda=True)
        else:
            st.write("Inference on CPU")
            age_predictor = load_age_model(use_cuda=False)

    st.title("Bone age prediction model")

    file = st.file_uploader("Upload An Image")

    if file:
        img = np.array(Image.open(file))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        st.image([img, vis], caption=["Input", "Predicted Mask"], width=300)

        ignore_mask = st.checkbox(
            "Bad mask (ignore for processing)",
            help="If the hand detection did not work properly, bone age prediction can be carried out without the hand extraction step",
        )

        sex = st.radio(f"Is the patient female or male?", options=["Female", "Male"],)

        if st.button("Predict"):
            sex = 1 if sex == "Male" else 0
            mask = None if ignore_mask else mask
            with st.spinner("performing age assessment"):
                age, stats = age_predictor(img, sex, mask)

            st.title(f"Predicted age:")
            st.title(f"{age:.2f} months ({age / 12:.2f} years )")

            st.write(stats.to_html(escape=False), unsafe_allow_html=True)
