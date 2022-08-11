"""
module for custom functions used for creating and training the model
"""
import logging
import os
from typing import Union, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd

from bone_age.effNet import EfficientNet


class Predictor:
    def __init__(self):
        self.models = {
            "masked_effnet_super_shallow_fancy_aug": EfficientModel(
                "efficientnet-b0",
                pretrained_path="./models/masked_effnet_super_shallow_fancy_aug.ckpt",
                load_dense=True,
            ).eval(),
            "masked_effnet_supShal_highRes_fancy_aug": EfficientModel(
                "efficientnet-b0",
                pretrained_path="./models/masked_effnet_supShal_highRes_fancy_aug.ckpt",
                load_dense=True,
            ).eval(),
            "masked_effnet-b4_shallow_pretr_fancy_aug": EfficientModel(
                "efficientnet-b4",
                pretrained_path="./models/masked_effnet-b4_shallow_pretr_fancy_aug.ckpt",
                load_dense=True,
            ).eval(),
        }
        with open("./bone_age/parameters.yml", "r") as stream:
            self.params = yaml.safe_load(stream)
        self.mask_crop_size = 1.15
        self.data_aug = self.get_inference_augmentation()
        self.data_aug_highRes = self.get_inference_augmentation(height=1024, width=1024)

    def __call__(self, image, male, mask=None) -> float:
        images = self._preprocess_image(image, mask)
        norm_image = images[0].to(torch.float32).unsqueeze(dim=0)
        high_res_image = images[1].to(torch.float32).unsqueeze(dim=0)
        male = torch.Tensor([[male]])

        y_hats = {}
        with torch.no_grad():
            for name, model in self.models.items():
                if "highRes" in name:
                    y_hat = model(high_res_image, male)
                else:
                    y_hat = model(norm_image, male)
                print(f"{name} : {y_hat.item()}")
                y_hat_cor = self.cor_prediction_bias_wrapper(y_hat, name)
                y_hats[name] = {"raw": y_hat.item(), "cor": y_hat_cor.item()}
        stats = pd.DataFrame(y_hats).T
        return stats.cor.mean(), stats

    def _preprocess_image(self, image, mask):
        if mask is not None:
            image = self._apply_mask(image, mask)
        else:
            image = (image / image.max() * 255).astype(np.uint8)
        images = [
            self.data_aug(image=image)["image"],
            self.data_aug_highRes(image=image)["image"],
        ]
        proc = []
        for image in images:
            image = image.to(torch.float32)
            m = image.mean()
            sd = image.std()
            image = (image - m) / sd
            proc.append(image)
        return proc

    def _apply_mask(self, image, mask) -> np.ndarray:
        """
        apply image and subtract min intensity (1st percentile) from the masked area
        """
        image = cv2.bitwise_and(image, image, mask=mask)
        m = np.percentile(image[image > 0], 1)
        image = cv2.subtract(image, m)  # no underflow
        image = self._crop_to_mask(image, mask)
        return image

    def _crop_to_mask(self, image, mask):
        """
        rotate and flip image, and crop to mask if specified
        """
        x = np.nonzero(np.max(mask, axis=0))
        xmin, xmax = (np.min(x), np.max(x) + 1)
        y = np.nonzero(np.max(mask, axis=1))
        ymin, ymax = (np.min(y), np.max(y) + 1)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width // 2
        y_center = ymin + height // 2

        size = max(height, width)
        size = round(size * self.mask_crop_size)

        xmin_new = x_center - size // 2
        xmax_new = x_center + size // 2
        ymin_new = y_center - size // 2
        ymax_new = y_center + size // 2

        top = abs(min(0, ymin_new))
        bottom = max(0, ymax_new - mask.shape[0])
        left = abs(min(0, xmin_new))
        right = max(0, xmax_new - mask.shape[1])

        out = cv2.copyMakeBorder(
            image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT
        )
        ymax_new += top
        ymin_new += top
        xmax_new += left
        xmin_new += left

        return out[ymin_new:ymax_new, xmin_new:xmax_new]

    @staticmethod
    def get_inference_augmentation(width=512, height=512, rotation_angle=0, flip=False):
        return A.Compose(
            [
                A.transforms.HorizontalFlip(p=flip),
                A.augmentations.geometric.transforms.Affine(
                    rotate=(rotation_angle, rotation_angle), p=1.0,
                ),
                A.augmentations.crops.transforms.RandomResizedCrop(
                    width, height, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                ),
                ToTensorV2(),
            ],
            p=1,
        )

    @staticmethod
    def cor_prediction_bias(yhat, slope, intercept):
        """corrects model predictions (yhat) for linear bias (defined by slope and intercept)"""
        return yhat - (yhat * slope + intercept)

    def cor_prediction_bias_wrapper(self, y_hat, ckp_path):
        y_hat = y_hat * self.params["age_sd"] + self.params["age_mean"]
        ckp_path = os.path.basename(ckp_path)
        slope = self.params[ckp_path]["slope"]
        intercept = self.params[ckp_path]["intercept"]
        return self.cor_prediction_bias(y_hat, slope, intercept)


class EfficientModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet-b0",
        dense_layers: List[int] = [1024, 1024, 512, 512],
        input_size: (int, int, int) = (1, 512, 512),
        pretrained_path: str = "",
        load_dense: bool = False,
        dropout_p: float = 0.2,
        act_type: str = "mem_eff",
        n_gender_dcs: float = 32,
    ):
        """
        Bone disorder model with efficientnet backbone

        optimized to load pretrained weights from bone age prediction models
        """
        super(EfficientModel, self).__init__()
        weight_dict = torch.load(pretrained_path, map_location="cpu")
        assert (
            backbone in EfficientNet.VALID_MODELS
        ), f"Given base model type ({backbone}) is invalid"
        if pretrained_path == "imagenet":
            assert (
                backbone != "efficientnet-l2"
            ), "'efficientnet-l2' does not currently have pretrained weights"
            self.base = EfficientNet.EfficientNet.from_pretrained(
                backbone, in_channels=input_size[0]
            )
        else:
            self.base = EfficientNet.EfficientNet.from_name(
                backbone, in_channels=input_size[0]
            )
            if pretrained_path:  # pretrained from path --> bone age model
                self.base.load_state_dict(self.load_weights(weight_dict, "base"))
        self.act = (
            EfficientNet.Swish()
            if act_type != "mem_eff"
            else EfficientNet.MemoryEfficientSwish()
        )
        self.dropout = nn.Dropout(p=dropout_p)

        self.dense_blocks = nn.ModuleList()
        if not load_dense:
            self.fc_male_in = nn.Linear(1, n_gender_dcs)
            features_dim = EfficientNet.FEATURE_DIMS[
                backbone
            ]  # 2nd dim of feature tensor
            channel_sizes_in = [features_dim + n_gender_dcs] + dense_layers

            for idx in range(len(channel_sizes_in) - 1):
                self.dense_blocks.append(
                    nn.Linear(
                        in_features=channel_sizes_in[idx],
                        out_features=channel_sizes_in[idx + 1],
                    )
                )
        else:
            d = self.load_weights(weight_dict, "fc_gender_in")
            self.fc_male_in = nn.Linear(1, d["weight"].shape[0])
            self.fc_male_in.load_state_dict(d)

            d = self.load_weights(weight_dict, "dense_blocks")
            for k, v in d.items():
                if "weight" in k:
                    self.dense_blocks.append(
                        nn.Linear(in_features=v.shape[1], out_features=v.shape[0])
                    )
            self.dense_blocks.load_state_dict(d)

        self.fc_disorder = nn.Linear(self.dense_blocks[-1].out_features, 1)
        if load_dense:
            self.fc_disorder.load_state_dict(
                self.load_weights(weight_dict, "fc_boneage")
            )
        self.base._fc = None

    def forward(self, x, male):
        x = self.base.extract_features(x, return_residual=False)
        x = torch.mean(x, dim=(2, 3))  # agnostic of the 3th and 4th dim (h,w)
        x = self.dropout(x)
        x = self.act(x)
        x = x.view(x.size(0), -1)

        male = self.act(self.fc_male_in(male))
        x = torch.cat([x, male], dim=-1)  # expected size = B x 1312

        for mod in self.dense_blocks:
            x = self.act(self.dropout(mod(x)))
        x = self.fc_disorder(x)
        return x

    @staticmethod
    def load_weights(weight_dict, key="base"):
        """
        load part specified by key from the models stored at the path
        """
        key += "."
        return {
            k.replace(key, ""): v
            for k, v in weight_dict["state_dict"].items()
            if key in k
        }
