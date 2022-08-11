"""
custom metrics
"""
import pandas as pd
import torch
import numpy as np
from scipy.special import softmax
from typing import Union, List


def softmax_confusion_matrix(
    y: Union[torch.Tensor, np.ndarray, pd.Series],
    y_hat: Union[torch.Tensor, np.ndarray],
    apply_softmax=True,
    class_names: List = [],
) -> np.ndarray:
    """
    Calculates the confusion matrix for logits tensor.

    In contrast to a usual multi-class conf matrix, this gives an idea about the average confidence the model has w.r.t. discriminating the classes
    Note, this follow the sklearn conventions (gt rows as columns)

    :param y_hat: model predictions (logits or softmax) shape [n, C] or [C]
    :param y: gt labels (shape [n])
    :param apply_softmax: whether softmax still need to be applied internally
    :param class_names: names to consider if y_hat is provided as str labels rather than numbers
    """
    n, c = y_hat.shape

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()

    if apply_softmax:
        y_hat = softmax(y_hat, axis=-1)

    conf_mat = np.zeros((c, c))
    if not class_names:
        y = y.astype(int)
        for k in range(n):
            conf_mat[y[k], :] += y_hat[k].reshape(1, -1)
    else:
        for k in range(n):
            if y[k] in class_names:
                k_hat = class_names.index(y[k])
                conf_mat[k_hat, :] += y_hat[k]
    return conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
