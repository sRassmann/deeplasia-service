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
from bone_age.models import (
    EfficientModel as BoneAgeModel,
    Predictor as AgePredictor,
    MultiTaskModel as SexModel,
    SexPredictor,
)


@st.cache()
def load_mask_model(
    path: str = "./models/fscnn_cos.ckpt", use_cuda=False
) -> MaskPredictor:
    return MaskPredictor(checkpoint=path, use_cuda=use_cuda)


@st.cache()
def load_age_model(use_cuda=False) -> AgePredictor:
    ensemble = {
        "masked_effnet_super_shallow_fancy_aug": BoneAgeModel(
            "efficientnet-b0",
            pretrained_path="./models/masked_effnet_super_shallow_fancy_aug.ckpt",
            load_dense=True,
        ).eval(),
        "masked_effnet_supShal_highRes_fancy_aug": BoneAgeModel(
            "efficientnet-b0",
            pretrained_path="./models/masked_effnet_supShal_highRes_fancy_aug.ckpt",
            load_dense=True,
        ).eval(),
        "masked_effnet-b4_shallow_pretr_fancy_aug": BoneAgeModel(
            "efficientnet-b4",
            pretrained_path="./models/masked_effnet-b4_shallow_pretr_fancy_aug.ckpt",
            load_dense=True,
        ).eval(),
    }
    return AgePredictor(ensemble, use_cuda=use_cuda)


@st.cache()
def load_sex_model(use_cuda=False) -> SexPredictor:
    ensemble = {
        "sex_model_mtl": SexModel.load_from_checkpoint(
            "./models/sex_pred_model.ckpt"
        ).eval()
    }
    return SexPredictor(ensemble, use_cuda=use_cuda)


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

    mask_predictor = load_mask_model()
    age_predictor = load_age_model()

    torch.set_num_threads(args.n_threads)

    if torch.cuda.is_available() and args.use_cuda:
        if st.checkbox("use GPU (experimental for prediction speed up)"):
            st.write("Inference on GPU (fast)")
            age_predictor = load_age_model(use_cuda=True)
            mask_predictor = load_mask_model(use_cuda=True)
        else:
            st.write("Inference on CPU (slow)")
            age_predictor = load_age_model(use_cuda=False)
            mask_predictor = load_mask_model(use_cuda=False)

    st.title("Bone age prediction model")

    file = st.file_uploader("Upload an image (png or jpg)")

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        mask = (mask > mask.max() // 2).astype(np.uint8)
        st.image([img, vis], caption=["Input", "Predicted Mask"], width=300)
        no_crop = st.checkbox(
            "Disable cropping",
            help="Disables cropping to the mask, might be advantageous for well-centered images like the orginal RSNA images ",
        )

        ignore_mask = st.checkbox(
            "Bad mask (ignore for processing)",
            help="If the hand detection did not work properly, bone age prediction can be carried out without the hand extraction step",
        )

        sex = st.radio(
            f"Is the patient female or male?",
            options=["Female", "Male", "Predict Sex"],
        )

        if st.button("Predict Bone Age"):
            mask = None if ignore_mask else mask

            if sex == "Predict Sex":
                sex_predictor = load_sex_model(use_cuda=False)
                with st.spinner("predicting sex"):
                    sex, _ = sex_predictor(
                        img, mask=mask, mask_crop=-1 if no_crop else 1.15
                    )
                sex = int(sex > 0.5)
                st.write(f"Predicted to be **{'male' if sex else 'female'}**")
            else:
                sex = 1 if sex == "Male" else 0

            with st.spinner("performing age assessment"):
                age, stats = age_predictor(
                    img, sex, mask=mask, mask_crop=-1 if no_crop else 1.15
                )

            st.title(f"Predicted age:")
            st.title(f"{age:.2f} months ({age / 12:.2f} years )")

            with st.expander("See details"):
                st.write(
                    stats.to_html(escape=False, float_format="{:20,.2f}".format),
                    unsafe_allow_html=True,
                )
