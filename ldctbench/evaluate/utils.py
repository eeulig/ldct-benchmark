import argparse
import importlib
import os
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import mean_squared_error, structural_similarity
from torchmetrics.functional.image import visual_information_fidelity

import ldctbench
from ldctbench.evaluate.ldct_iqa import LDCTIQA
from ldctbench.hub.methods import Methods
from ldctbench.utils import apply_center_width, load_json, load_yaml

package_dir = os.path.join(os.path.dirname(os.path.abspath(ldctbench.__file__)))

DATA_RANGE = 2924.0  # clip to maximum HU of bone (1900) + offset (1024) -> 2924
CHECKPOINTS = load_json(os.path.join(package_dir, "hub", "checkpoints.json"))
DATA_INFO = load_yaml(os.path.join(package_dir, "data", "info.yml"))

CW = {
    "C": (1024 - 600, 1500),
    "L": (1024 + 50, 400),
    "N": (1024 + 40, 80),
}  # Center width in HU + offset (1024)


def normalize(
    x: Union[torch.Tensor, np.ndarray],
    method: Union[
        Literal[
            Methods.RESNET,
            Methods.CNN10,
            Methods.DUGAN,
            Methods.QAE,
            Methods.REDCNN,
            Methods.TRANSCT,
            Methods.WGANVGG,
            Methods.BILATERAL,
        ],
        str,
        None,
    ] = None,
    normalization: Optional[str] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize tensor or ndarray based on normalization type of trained model.

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray to normalize
    method : Union[ Literal[ Methods.RESNET, Methods.CNN10, Methods.DUGAN, Methods.QAE, Methods.REDCNN, Methods.TRANSCT, Methods.WGANVGG, Methods.BILATERAL, ], str, None, ], optional
        Enum item or string, specifying model to dermine which normalization to use. See [ldctbench.hub.methods.Methods][] for more info, by default None
    normalization : Optional[str], optional
        Normalization method, must be meanstd | minmax, by default None

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Normalized tensor or ndarray

    Raises
    ------
    ValueError
        If `normalization` is neither "meanstd" nor "minmax"
    """
    if not (method or normalization):
        warnings.warn(
            "No method_name or normalization provided. Falling back to meanstd instead!"
        )
        normalization = "meanstd"
    elif method:
        if isinstance(method, str):
            method = Methods[method.upper()]
        normalization = CHECKPOINTS[method.value]["normalization"]

    if normalization == "meanstd":
        return (x - float(DATA_INFO["mean"])) / float(DATA_INFO["std"])
    elif normalization == "minmax":
        return (x - float(DATA_INFO["min"])) / (
            float(DATA_INFO["max"]) - float(DATA_INFO["min"])
        )
    else:
        raise ValueError(f"Unknown normalization method: {normalization}!")


def denormalize(
    x: Union[torch.Tensor, np.ndarray],
    method: Union[
        Literal[
            Methods.RESNET,
            Methods.CNN10,
            Methods.DUGAN,
            Methods.QAE,
            Methods.REDCNN,
            Methods.TRANSCT,
            Methods.WGANVGG,
            Methods.BILATERAL,
        ],
        str,
        None,
    ] = None,
    normalization: Optional[str] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize tensor or ndarray based on normalization type of trained model.

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray to normalize
    method : Union[ Literal[ Methods.RESNET, Methods.CNN10, Methods.DUGAN, Methods.QAE, Methods.REDCNN, Methods.TRANSCT, Methods.WGANVGG, Methods.BILATERAL, ], str, None, ], optional
        Enum item or string, specifying model to dermine which normalization to use. See [ldctbench.hub.methods.Methods][] for more info, by default None
    normalization : Optional[str], optional
        Normalization method, must be meanstd | minmax, by default None

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Normalized tensor or ndarray

    Raises
    ------
    ValueError
        If `normalization` is neither "meanstd" nor "minmax"
    """
    if not (method or normalization):
        warnings.warn(
            f"No method_name or normalization provided. Falling back to meanstd instead!"
        )
        normalization = "meanstd"
    elif method:
        if isinstance(method, str):
            method = Methods[method.upper()]
        normalization = CHECKPOINTS[method.value]["normalization"]

    if normalization == "meanstd":
        return x * float(DATA_INFO["std"]) + float(DATA_INFO["std"])
    elif normalization == "minmax":
        return x * (float(DATA_INFO["max"]) - float(DATA_INFO["min"])) + float(
            DATA_INFO["min"]
        )
    else:
        raise ValueError(f"Unknown normalization method: {normalization}!")


def preprocess(
    x: Union[torch.Tensor, np.ndarray],
    **normalization_kwargs,
) -> torch.Tensor:
    """Preprocess input tensor

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray]
        Input tensor or ndarray

    Returns
    -------
    torch.Tensor
        Preprocessed tensor
    """
    x = torch.from_numpy(x.astype("float32"))
    x = normalize(x, **normalization_kwargs)
    return torch.unsqueeze(x, 0)


def vif(x, y, sigma_n_sq=2.0):
    """Compute visual information fidelity"""
    x_t = torch.from_numpy(x.copy())
    y_t = torch.from_numpy(y.copy())
    return visual_information_fidelity(
        torch.unsqueeze(torch.unsqueeze(x_t, 0), 0),
        torch.unsqueeze(torch.unsqueeze(y_t, 0), 0),
        sigma_n_sq=sigma_n_sq,
    ).numpy()


def compute_metric(
    targets: Union[torch.Tensor, np.ndarray],
    predictions: Union[torch.Tensor, np.ndarray],
    metrics: List[str],
    denormalize_fn: Optional[Callable] = None,
    exam_type: Optional[str] = None,
    ldct_iqa: Optional[LDCTIQA] = None,
) -> Dict[str, List]:
    """Compute metrics for given predictions and targets

    Parameters
    ----------
    targets : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray of shape (mbs, 1, H, W) holding ground truth
    predictions : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray of shape (mbs, 1, H, W) holding predictions
    metrics : List[str]
        List of metrics must be "VIF" | "PSNR" | "RMSE" | "SSIM" | "LDCTIQA"
    denormalize_fn: Optional[Callable], optional
        Function to use for denormalizing, by default None
    exam_type : Optional[str], optional
        Exam type (for computing SSIM and PSNR on windowed images), by default None
    ldct_iqa : Optional[LDCTIQA], optional
        LDCTIQA object for computing LDCTIQA score, by default None

    Returns
    -------
    Dict[str, List]
        Dictionary containing for each metric a list of len(mbs)

    Raises
    ------
    ValueError
        If `predictions.shape != targets.shape`
    ValueError
        If element in metric is not "VIF" | "PSNR" | "RMSE" | "SSIM" | "LDCTIQA"
    """
    if isinstance(targets, torch.Tensor):
        if targets.is_cuda:
            targets = targets.data.cpu().numpy()
        else:
            targets = targets.data.numpy()
    if isinstance(predictions, torch.Tensor):
        if predictions.is_cuda:
            predictions = predictions.data.cpu().numpy()
        else:
            predictions = predictions.data.numpy()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape of predictions and targets must be identical but got {predictions.shape} and {targets.shape}!"
        )

    if denormalize_fn:
        targets = denormalize_fn(targets)
        predictions = denormalize_fn(predictions)

    n_imgs = predictions.shape[0]
    res = {}
    for metric in metrics:
        res[metric] = []
        for im_idx in range(n_imgs):
            t = np.copy(targets[im_idx]).squeeze()
            p = np.copy(predictions[im_idx]).squeeze()
            if metric == "VIF":
                t = np.clip(t, a_min=0.0, a_max=DATA_RANGE)
                p = np.clip(p, a_min=0.0, a_max=DATA_RANGE)
                res[metric].append(vif(t, p).item())
            elif metric == "SSIM":
                t = apply_center_width(
                    t, center=CW[exam_type][0], width=CW[exam_type][1]
                )
                p = apply_center_width(
                    p, center=CW[exam_type][0], width=CW[exam_type][1]
                )
                res[metric].append(structural_similarity(t, p, data_range=1.0).item())
            elif metric == "PSNR":
                t = apply_center_width(
                    t, center=CW[exam_type][0], width=CW[exam_type][1]
                )
                p = apply_center_width(
                    p, center=CW[exam_type][0], width=CW[exam_type][1]
                )
                mse = mean_squared_error(t, p)
                res[metric].append(10 * np.log10((1.0**2) / mse).item())
            elif metric == "RMSE":
                t = np.clip(t, a_min=0.0, a_max=DATA_RANGE)
                p = np.clip(p, a_min=0.0, a_max=DATA_RANGE)
                res[metric].append(np.sqrt(mean_squared_error(t, p)).item())
            elif metric == "LDCTIQA":
                # This is a no-reference metric, thus only needs predictions
                res[metric].append(ldct_iqa(p).item())
            else:
                raise ValueError(f"Unknown metric {metric}")
    return res


def setup_trained_model(
    run_name: str,
    device: torch.device = torch.device("cuda"),
    network_name: str = "Model",
    state_dict: str = None,
    return_args: bool = False,
    return_model: bool = True,
    eval: bool = True,
    **model_kwargs,
) -> Union[Tuple[nn.Module, argparse.Namespace], nn.Module, argparse.Namespace]:
    """Setup a trained model with run in `./wandb`

    Parameters
    ----------
    run_name : str
        Name of run (is the same as foldername)
    device : torch.device, optional
        Device to move model to, by default torch.device("cuda")
    network_name : str, optional
        Class name of network, by default "Model"
    state_dict : str, optional
        Name of state_dict. If None, model is initialized with random parameters, by default None
    return_args : bool, optional
        Return args of training run, by default False
    return_model : bool, optional
        Return model, by default True
    eval : bool, optional
        Set model to eval mode, by default True

    Returns
    -------
    Union[Tuple[nn.Module, argparse.Namespace], nn.Module, argparse.Namespace]
        Either model, args or model and args of training run
    """
    savepath = os.path.join("wandb", run_name, "files")
    args = argparse.Namespace(**load_yaml(os.path.join(savepath, "args.yaml")))
    if return_args and not return_model:
        return args
    model_class = getattr(
        importlib.import_module("ldctbench.methods.{}.network".format(args.trainer)),
        network_name,
    )
    model_args = args
    model = model_class(model_args, **model_kwargs).to(device)
    if state_dict:
        state = torch.load(
            os.path.join(savepath, "{}.pt".format(state_dict)), weights_only=True
        )
        print(
            f"Restore state dict {state_dict} of {network_name} from iteration",
            state["iteration"],
        )
        model.load_state_dict(state["model_state_dict"])
    if eval:
        model.eval()
    if return_args:
        return model, args
    return model


def save_raw(filepath, filename, X):
    X.tofile(
        os.path.join(
            filepath,
            filename
            + "_"
            + str(X.shape[1])
            + "x"
            + str(X.shape[2])
            + "x"
            + str(X.shape[0])
            + ".raw",
        )
    )
