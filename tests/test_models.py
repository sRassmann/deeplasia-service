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


def main(use_cuda=True):
    mask_predictor = MaskPredictor(checkpoint="./models/fscnn_cos.ckpt")
    age_predictor = AgePredictor(use_cuda=use_cuda)

    dir = "/home/rassman/bone2gene/data/annotated/rsna_bone_age/bone_age_test_data_set"

    d = {}
    for i, path in enumerate(sorted(os.listdir(dir))):
        img = Image.open(os.path.join(dir, path))

        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        sex = i < 100

        age, stats = age_predictor(img, sex, mask)

        d[path] = (
            {"sex": "M" if sex else "F", "pred": age}
            | {f"cor_{k}": v for k, v in stats.cor.to_dict().items()}
            | {f"raw_{k}": v for k, v in stats.raw.to_dict().items()}
        )

    df = pd.DataFrame(d).T
    gt = pd.read_csv(
        "/home/rassman/bone2gene/data/annotated/rsna_bone_age/annotation_bone_age_test_data_set.csv"
    )
    gt["Case ID"] = gt["Case ID"].astype(str) + ".png"
    df = gt.merge(df, left_on="Case ID", right_index=True)
    df.to_csv("tests/result.csv")

    print(
        np.linalg.norm(df["Ground truth bone age (months)"] - df["pred"], 1) / len(df)
    )


if __name__ == "__main__":
    main()
