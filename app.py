from flask import Flask, jsonify, request
import torch

import numpy as np
import cv2

from fscnn.predict import Predictor as MaskPredictor
from bone_age.models import (
    EfficientModel as BoneAgeModel,
    Predictor as AgePredictor,
    MultiTaskModel as SexModel,
    SexPredictor,
)

app = Flask(__name__)

use_cuda = torch.cuda.is_available()
enable_sex_prediction = True
threads = 4

mask_model_path = "./models/fscnn_cos.ckpt"
ensemble = {
    "masked_effnet_super_shallow_fancy_aug": BoneAgeModel(
        "efficientnet-b0",
        pretrained_path="./models/masked_effnet_super_shallow_fancy_aug.ckpt",
        load_dense=True,
    ).eval(),
    # "masked_effnet_supShal_highRes_fancy_aug": BoneAgeModel(
    #     "efficientnet-b0",
    #     pretrained_path="./models/masked_effnet_supShal_highRes_fancy_aug.ckpt",
    #     load_dense=True,
    # ).eval(),
    # "masked_effnet-b4_shallow_pretr_fancy_aug": BoneAgeModel(
    #     "efficientnet-b4",
    #     pretrained_path="./models/masked_effnet-b4_shallow_pretr_fancy_aug.ckpt",
    #     load_dense=True,
    # ).eval(),
}
if enable_sex_prediction:
    sex_model_ensemble = {
        "sex_model_mtl": SexModel.load_from_checkpoint(
            "./models/sex_pred_model.ckpt"
        ).eval()
    }

torch.set_num_threads(threads)
mask_predictor = MaskPredictor(checkpoint=mask_model_path, use_cuda=use_cuda)
age_predictor = AgePredictor(ensemble, use_cuda=use_cuda)
sex_predictor = (
    SexPredictor(ensemble, use_cuda=use_cuda) if enable_sex_prediction else None
)


def get_prediction(image_bytes, sex, use_mask):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    sex_predicted = False

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if use_mask:
        try:
            mask, vis = mask_predictor(img)
        except Exception as e:
            print("no mask found")
            mask = np.ones_like(img)
            vis = img.copy()
    else:
        mask = np.ones_like(img)
        vis = img.copy()
    mask = (mask > mask.max() // 2).astype(np.uint8)

    if sex in ["Male", "male", "m", "M"]:
        sex, sex_input = "m", 1
    elif sex in ["Female", "female", "f", "F"]:
        sex, sex_input = "f", 0

    if sex not in ["m", "f"]:
        if sex_predictor is not None:
            sex, _ = sex_predictor(img, mask=mask, mask_crop=1.15)
            sex_input = sex > 0.5
            sex = "m" if sex else "f"
            sex_predicted = True
        else:
            raise Exception("Sex is not provided and sex inference disabled")

    age, stats = age_predictor(img, sex_input, mask=mask, mask_crop=1.15)

    return age.item(), sex, sex_predicted


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            print("no file")
            return jsonify({"error": "No file part"})
        file = request.files["file"]
        image_bytes = file.read()

        sex = request.form.get("sex")
        use_mask = request.form.get("use_mask")

        bone_age, sex, sex_predicted = get_prediction(image_bytes, sex, use_mask)
        return jsonify(
            {
                "bone_age": bone_age,
                "used_sex": sex,
                "sex_predicted": sex_predicted,
            }
        )


if __name__ == "__main__":
    app.run()


# with open("../data/public/Achondroplasia_Slide6.PNG", "rb") as f:
#     image_bytes = f.read()
#     print(get_prediction(image_bytes))

# import requests

# url = "http://localhost:5000/predict"

# test_img = "/home/sebastian/bone2gene/data/public/Achondroplasia_Slide6.PNG"
# files = {'file': open(test_img,'rb')}

# data = {
#     "sex": "female",
#     "use_mask": "1"  # 1 for True, 0 for False
# }

# resp = requests.post(url, files=files, data=data)
# resp.json()
