import cv2
from PIL import Image
import os
import torch

import streamlit as st
import pandas as pd
import numpy as np

from tqdm import tqdm

from fscnn.lib.models import MaskModel
from fscnn.predict import Predictor as MaskPredictor
from bone_age.models import Predictor as AgePredictor


def main(use_cuda=True):
    mask_predictor = MaskPredictor(checkpoint="./models/fscnn_cos.ckpt")
    age_predictor = AgePredictor(use_cuda=use_cuda)

    print("running test on RSNA test data set")
    dir = "/home/rassman/bone2gene/data/annotated/rsna_bone_age/bone_age_test_data_set"
    d = {}
    for i, path in tqdm(enumerate(sorted(os.listdir(dir)))):
        img = Image.open(os.path.join(dir, path))

        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        mask = (mask > mask.max() // 2).astype(np.uint8)

        sex = i < 100

        age, stats = age_predictor(img, sex, mask=mask, mask_crop=-1)

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
    df.to_csv("tests/result_rsna.csv")

    print(
        np.linalg.norm(df["Ground truth bone age (months)"] - df["pred"], 1) / len(df)
    )

    print("running test on Bonn test data set")
    dir = "/home/rassman/bone2gene/data/annotated/UKB_all"
    d = {}
    for i, path in tqdm(enumerate(sorted(os.listdir(dir)))):
        img = Image.open(os.path.join(dir, path))

        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, vis = mask_predictor(img)
        mask = (mask > mask.max() // 2).astype(np.uint8)

        sex = "M" in path.upper()

        age, stats = age_predictor(img, sex, mask=mask, mask_crop=1.15)

        d[path] = (
            {"sex": "M" if sex else "F", "pred": age}
            | {f"cor_{k}": v for k, v in stats.cor.to_dict().items()}
            | {f"raw_{k}": v for k, v in stats.raw.to_dict().items()}
        )

    df = pd.DataFrame(d).T
    gt = pd.read_csv(
        "/home/rassman/bone2gene/bone-age/nb/paper/born/predictions_all.csv"
    )
    gt.image_ID = gt.image_ID.str.lower()
    df.index = df.index.str.lower()
    df = gt.merge(df, left_on="image_ID", right_index=True)
    df.to_csv("tests/result_bonn.csv")

    print(
        f"Score wrt to batches predictions : {np.linalg.norm(df['ensembled'] - df['pred'], 1) / len(df)}"
    )

    df = df.dropna()
    print(
        f"Score wrt to gt : {np.linalg.norm(df['bone_age'] - df['pred'], 1) / len(df)}"
    )


if __name__ == "__main__":
    main()
