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
import argparse

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
    parser = argparse.ArgumentParser(description="This app lists animals")
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Number of threads to run inference with",
    )
    parser.add_argument(
        "--use_cuda", action="store_true", help="Try to use GPU (is available)"
    )
    args = parser.parse_args()

    mask_predictor = load_fscnn()
    age_predictor = load_age_model()

    torch.set_num_threads(args.n_threads)

    if torch.cuda.is_available() and args.use_cuda:
        if st.checkbox("use GPU"):
            st.write("Inference on GPU")
            age_predictor = load_age_model(use_cuda=True)
        else:
            st.write("Inference on CPU")
            age_predictor = load_age_model(use_cuda=False)

    st.title("Bone age prediction model")

    file = st.file_uploader("Upload An Image")

    if file:
        no_crop = st.checkbox("disable cropping")

        img = np.array(Image.open(file))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        mask = (mask > mask.max() // 2).astype(np.uint8)
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
                age, stats = age_predictor(
                    img, sex, mask=mask, mask_crop=-1 if no_crop else 1.15
                )

            st.title(f"Predicted age:")
            st.title(f"{age:.2f} months ({age / 12:.2f} years )")

            st.write(stats.to_html(escape=False), unsafe_allow_html=True)
