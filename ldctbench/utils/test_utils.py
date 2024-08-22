import argparse
import importlib
import os
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import offsetbox
from skimage.metrics import mean_squared_error, structural_similarity
from torchmetrics.functional.image import visual_information_fidelity

import ldctbench
from ldctbench.hub.methods import Methods
from ldctbench.utils import load_json, load_yaml

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


def apply_center_width(
    x: np.ndarray,
    center: Union[int, float],
    width: Union[int, float],
    out_range: Tuple = (0, 1),
) -> np.ndarray:
    """Apply some center and width to an image

    Parameters
    ----------
    x : np.ndarray
        Image array of arbitrary dimension
    center : float
        Float indicating the center
    width : float
        Float indicating the width
    out_range : Tuple, optional
        Desired output range, by default (0, 1)

    Returns
    -------
    np.ndarray
        Copy of input array with center and width applied
    """
    center = float(center)
    width = float(width)
    lower = center - 0.5 - (width - 1) / 2.0
    upper = center - 0.5 + (width - 1) / 2.0
    res = np.empty(x.shape, dtype=x.dtype)
    res[x <= lower] = out_range[0]
    res[(x > lower) & (x <= upper)] = (
        (x[(x > lower) & (x <= upper)] - (center - 0.5)) / (width - 1.0) + 0.5
    ) * (out_range[1] - out_range[0]) + out_range[0]
    res[x > upper] = out_range[1]
    return res


def add_crop(
    ax: plt.Axes,
    image: np.array,
    coords: Tuple[int, int],
    cropsize: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
    edgecolor: Tuple[float] = (1.0, 0.0, 0.0),
    pos_crop: Tuple[int, int] = (0.0, 0.0),
    lw_crop: float = 1.0,
    lw_overlay: float = 1.0,
    **kwargs,
):
    """Function to add crop to an existing matplotlib axis

    Parameters
    ----------
    ax : plt.Axes
        Axis to add plot to
    image : np.array
        2-dim array containing the (uncropped) image
    coords : Tuple[int, int]
        Center coordinates of the crop
    cropsize : int
        Cropsize in pixels
    vmin : float, optional
        vmin to use for the cropped image, by default 0.0
    vmax : float, optional
        vmax to use for the cropped image, by default 1.0
    zoom : float, optional
        Zoom of the cropped image, by default 1.0
    edgecolor : Tuple[float], optional
        Edgecolor around crop and bbox, by default (1.,0.,0.)
    pos_crop : Tuple[int, int], optional
        Position (in pixels) to place crop on the axis, by default (0., 0.)
    lw_crop : float, optional
        Linewidth of border around crop, by default 1.
    lw_overlay : float, optional
        Linewidth of border in uncropped image on axis, by default 1.
    kwargs: Additional kwargs passed to offsetbox.OffsetImage
        (e.g., `cmap`, `zoom`)

    Examples
    --------

    >>> im = np.random.uniform(low=24.0, high=4096.0, size=(128, 128))
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(im, vmin=0., vmax=2000., cmap="gray")
    >>> add_crop(ax, im, coords=(64,64), cropsize=32, vmin=0., vmax=2000., cmap='gray', pos_crop=(0,0))
    >>> plt.show()

    """
    x, y = coords
    crop_half = cropsize // 2
    im = image[x - crop_half : x + crop_half, y - crop_half : y + crop_half]
    im = offsetbox.OffsetImage(im, **kwargs)
    img = im.get_children()[0]
    img.set_clim(vmin=vmin, vmax=vmax)
    im.image.axes = ax
    ab = offsetbox.AnnotationBbox(
        im,
        (0, 0),
        xycoords="data",
        xybox=pos_crop,
        pad=0.0,
        bboxprops=dict(edgecolor=edgecolor, linewidth=lw_crop),
    )
    ax.add_artist(ab)

    # Add rectangular patch to image
    rect = patches.Rectangle(
        (coords[1] - crop_half, coords[0] - crop_half),
        cropsize,
        cropsize,
        linewidth=lw_overlay,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)


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
) -> Dict[str, List]:
    """Compute metrics for given predictions and targets

    Parameters
    ----------
    targets : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray of shape (mbs, 1, H, W) holding ground truth
    predictions : Union[torch.Tensor, np.ndarray]
        Tensor or ndarray of shape (mbs, 1, H, W) holding predictions
    metrics : List[str]
        List of metrics must be "vif" | "psnr" | "rmse" | "ssim"
    denormalize_fn: Optional[Callable], optional
        Function to use for denormalizing, by default None
    exam_type : Optional[str], optional
        Exam type (for computing SSIM and PSNR on windowed images), by default None

    Returns
    -------
    Dict[str, List]
        Dictionary containing for each metric a list of len(mbs)

    Raises
    ------
    ValueError
        If `predictions.shape != targets.shape`
    ValueError
        If element in metric is not "vif" | "psnr" | "rmse" | "ssim"
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
