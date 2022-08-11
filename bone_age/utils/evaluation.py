"""
utilities to run the evaluation of trained models
"""
import os
import shutil
from glob import glob
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from lib.utils.visualize import save_confusion_matrix
from lib.utils.metrics import softmax_confusion_matrix

import logging

logger = logging.getLogger(__name__)


class Evaluator:
    RELEVANT_ENTRIES = [  # metadata that is kept for the merged output df
        "image_ID",
        "patient_ID",
        "sex",
        "chronological_age",
        "bone_age",
        "image_source",
        "disorder",
    ]

    def __init__(
        self,
        model,
        trainer,
        loaders,
        anno_df,
        output_dir,
        name=None,
        checkpoint_path="path/to/model",
    ):
        self.model = model
        self.ckp_path = checkpoint_path
        self.name = name if name else self.model.name

        self.trainer = trainer
        self.loaders = loaders
        self.anno_df = pd.read_csv(anno_df) if os.path.exists(anno_df) else ""
        self.base_dir = os.path.join(output_dir, "output")
        self.merged_df = {}
        self.losses = {}
        self.y_col = ""

        os.makedirs(self.base_dir, exist_ok=True)

    def test_model(self):
        """
        Run test loop on all loaders
        """
        for name, loader in self.loaders.items():
            if not loader.dataset:
                continue
            self.y_col = loader.dataset.y_col
            output_path = os.path.join(self.base_dir, name, "predictions")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)

            raw_output = os.path.join(
                output_path, f"raw_predictions_{self.name}_{name}.csv"
            )
            merged_output = os.path.join(
                output_path, f"predictions_and_annotations_{self.name}_{name}.csv"
            )

            outputs = self.trainer.predict(model=self.model, dataloaders=loader)
            merged_df = self._write_predictions(outputs, raw_output, merged_output)
            if merged_df is not None:
                self.merged_df[name] = merged_df

            loss = (
                torch.mean(torch.concat([o["y"] for o in outputs])).item()
                if "loss" in outputs[0].keys()
                else None
            )
            if loss:
                self.losses[name] = loss

    def evaluate(self):
        for set_type, df in self.merged_df.items():
            if self.y_col in df.columns:
                output_path = os.path.join(
                    self.base_dir, set_type, "confusion_matrices"
                )
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                os.makedirs(output_path)
                self._save_conf_matrices(
                    df, os.path.join(output_path), set_type,
                )

                output_path = os.path.join(self.base_dir, set_type, "metrics")
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                os.makedirs(output_path)
                results = self._write_scores(df, output_path, set_type)
                if set_type in ["val", "test"]:
                    self.trainer.logger.log_metrics(
                        {
                            f"hp/{set_type}_loss": self.losses[set_type],
                            f"hp/{set_type}_accuracy": results["accuracy"][
                                "weighted avg"
                            ],
                            f"hp/{set_type}_macro_f1": results["f1-score"]["macro avg"],
                            f"hp/{set_type}_weighted_f1": results["f1-score"][
                                "weighted avg"
                            ],
                        }
                        | {
                            f"hp/{set_type}_f1_{name}": results["f1-score"][name]
                            for name in self.model.class_names
                            if name in results["f1-score"].keys()
                        }
                    )

    def _write_scores(self, df, output_path, set_type="test"):
        df = df.loc[df[self.y_col].isin(self.model.class_names)]
        rep = classification_report(
            df[self.y_col], df.prediction, digits=4, zero_division=0
        )
        with open(os.path.join(output_path, f"report_{set_type}.txt"), "w+") as f:
            f.write(f"Results for {self.name} ({self.ckp_path}) on {set_type}\n")
            if set_type in self.losses:
                f.write(f"avg loss: {self.losses[set_type]}\n")
            f.write(rep)
        logger.info(f"{'='*10} Classification results on {set_type} {'='*10}:\n{rep}")
        rep_dict = classification_report(
            df[self.y_col], df.prediction, output_dict=True, zero_division=0
        )
        weigh_acc = rep_dict.pop("accuracy")
        rep = pd.DataFrame(rep_dict).T
        acc = np.diag(confusion_matrix(df[self.y_col], df.prediction, normalize="true"))
        av_acc = np.mean(acc)
        rep.insert(1, "accuracy", np.append(acc, [av_acc, weigh_acc]))
        rep.to_csv(os.path.join(output_path, f"report_{set_type}.csv"))
        return rep

    def _save_conf_matrices(self, df, output_path, set_name="test"):
        y, y_hat = df[self.y_col], df["prediction"]

        self._save_conf_matrix_image_and_values(
            confusion_matrix(y, y_hat, labels=self.model.class_names, normalize="true"),
            self.model.class_names,
            output_path,
            f"{self.name}_{set_name}",
        )
        self._save_conf_matrix_image_and_values(
            confusion_matrix(y, y_hat, labels=y.unique(), normalize="true"),
            list(y.unique()),
            output_path,
            f"all_{self.name}_{set_name}",
        )
        self._save_conf_matrix_image_and_values(
            confusion_matrix(y, y_hat, labels=self.model.class_names),
            self.model.class_names,
            output_path,
            f"unnorm_{self.name}_{set_name}",
        )
        self._save_conf_matrix_image_and_values(
            confusion_matrix(y, y_hat, labels=y.unique()),
            list(y.unique()),
            output_path,
            f"all_unnorm_{self.name}_{set_name}",
        )

        # filter for first visit
        tmp = df.copy()
        tmp["image_ID"] = tmp["image_ID"].str.replace(r"\d", "", regex=True)
        tmp = tmp.drop_duplicates(["image_ID", "patient_ID"])
        self._save_conf_matrix_image_and_values(
            confusion_matrix(
                tmp[self.y_col],
                tmp.prediction,
                labels=self.model.class_names,
                normalize="true",
            ),
            self.model.class_names,
            output_path,
            f"first_visit_{self.name}_{set_name}",
        )

        sf = np.array(df[self.model.class_names])
        sf_conf_mat = softmax_confusion_matrix(
            y, sf, apply_softmax=False, class_names=list(self.model.class_names)
        )
        self._save_conf_matrix_image_and_values(
            sf_conf_mat,
            self.model.class_names,
            output_path,
            f"softmax_{self.name}_{set_name}",
        )

    @staticmethod
    def _save_conf_matrix_image_and_values(conf_mat, class_names, output_path, name):
        save_confusion_matrix(
            conf_mat, os.path.join(output_path, f"conf_mat_{name}.png"), class_names,
        )
        d = pd.DataFrame(conf_mat)
        d.columns = class_names
        d.index = class_names
        d.to_csv(os.path.join(output_path, f"raw_data_{name}.csv"))

    def _write_predictions(self, outputs, raw_output_path, merged_output_path):
        y = (
            torch.concat([o["y"] for o in outputs])
            if "y" in outputs[0].keys()
            else None
        )
        softmax = torch.concat([o["y_hat"] for o in outputs])
        names = [
            val.split("/")[-1]
            for sublist in [o["image_path"] for o in outputs]
            for val in sublist
        ]
        if self.model.class_names == ["bone_age"]:
            out = os.path.join(
                self.base_dir.replace("/output", ""),
                self.ckp_path.split("/")[-1] + ".csv",
            )
            pd.DataFrame({"image_ID": names, "y_hat": softmax.squeeze()}).to_csv(out)
            logger.info(f"saved to {out}")
            import shutil

            shutil.rmtree(self.base_dir)
            return None

        y_hat = pd.DataFrame(
            {"image_ID": names}
            | {l: softmax[:, i] for i, l in enumerate(self.model.class_names)}
        )
        y_hat.to_csv(raw_output_path)
        logger.info(f"raw predictions saved to {raw_output_path}")

        pred_label = torch.max(softmax, dim=1)[1]
        pred_label = [self.model.class_names[l] for l in pred_label]
        y_hat.insert(1, "prediction", pred_label)

        if isinstance(self.anno_df, pd.DataFrame):
            if y is not None:
                y_hat.insert(1, "y", y.to(int))
            merged = pd.merge(self.anno_df, y_hat, on="image_ID")
            if y is not None:
                assert np.all(
                    np.array([self.model.class_names[y] for y in merged.y])
                    == merged[self.y_col]
                )
            df_columns = (
                [c for c in self.RELEVANT_ENTRIES if c in merged.columns]
                + ["prediction"]
                + self.model.class_names
            )
            merged = merged[df_columns].sort_values(
                ["disorder", "patient_ID", "chronological_age"]
            )
            merged.to_csv(merged_output_path)
            return merged
        else:
            y_hat.to_csv(merged_output_path)
            return None

    @staticmethod
    def run_evaluation(
        model,
        trainer,
        loaders: dict,
        anno_df="data-management/annotation.csv",
        output_dir="output/predictions/",
        name=None,
        ckp_path="path/to/model",
    ):
        ev = Evaluator(model, trainer, loaders, anno_df, output_dir, name, ckp_path)
        ev.test_model()
        ev.evaluate()
