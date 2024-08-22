import os
import warnings
from typing import Literal, Optional, Union

import numpy as np
import pydicom
import torch
from tqdm import tqdm

from ldctbench.hub import Methods, load_model
from ldctbench.utils.test_utils import denormalize, normalize


@torch.no_grad()
def denoise_numpy(
    x: np.ndarray,
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
        torch.nn.Module,
    ],
    device: Optional[torch.device] = None,
    num_slices_parallel: int = 6,
    is_normalized=False,
) -> np.ndarray:
    """Denoise a numpy array with a model from the model hub

    Parameters
    ----------
    x : np.ndarray
        Input numpy array. Must be either 2d (single slice) or 3d (with slices-first) order
    method : Union[ Literal[ Methods.RESNET, Methods.CNN10, Methods.DUGAN, Methods.QAE, Methods.REDCNN, Methods.TRANSCT, Methods.WGANVGG, Methods.BILATERAL, ], str, torch.nn.Module, ]
        Enum item or string, specifying model to load. See [ldctbench.hub.methods.Methods][] for more info. You can also pass a nn.Module directly.
    device : Optional[torch.device], optional
        torch.device, optional
    num_slices_parallel : int, optional
        Number of slices to process in parallel, by default 6
    is_normalized : bool, optional
        Whether data in array is already normalized for the model, by default False

    Returns
    -------
    np.ndarray
        Prediction of the network

    Raises
    ------
    ValueError
        If input is not a 2d or 3d array
    """

    Nd = x.ndim
    if not 2 <= Nd <= 3:
        raise ValueError(
            f"Input must be 2 dimensional (single slice) or 3 dimensional (with slices-first) order. Got {x.ndim}d array of shape {x.shape} instead!"
        )

    if Nd == 2:
        x = np.expand_dims(x, axis=0)

    # If no device given, try cuda, else use cpu
    if not device:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # If method is not a nn.Module already, load the network
    if not isinstance(method, torch.nn.Module):
        method = load_model(method=method, device=device)

    # Preprocess inputs
    inputs = torch.from_numpy(x.astype("float32"))
    if not is_normalized:
        inputs = normalize(inputs, normalization="meanstd")

    # Allocate tensor to hold result
    predictions = torch.zeros_like(inputs, dtype=torch.float32)

    # Batch the input for parallel processing
    inputs_batched = torch.split(inputs, num_slices_parallel)

    # Run inference on batched input
    for i in range(len(inputs_batched)):
        this_input = torch.unsqueeze(inputs_batched[i], 1).to(
            dtype=torch.float32, device=device
        )
        pred = method(this_input)
        predictions[i * num_slices_parallel : (i + 1) * num_slices_parallel] = (
            pred.detach().cpu().squeeze()
        )

    # Denormalize if input was also not normalized
    if not is_normalized:
        res = denormalize(predictions, normalization="meanstd").numpy()
    else:
        res = predictions.numpy()

    if Nd == 2:
        return res.squeeze()
    return res


def denoise_dicom(
    dicom_path: str,
    savedir: str,
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
    ],
    device: Optional[torch.device] = None,
    disable_progress: bool = False,
):
    """Denoise a single DICOM or multiple DICOMs in a folder with a model from the model hub

    Parameters
    ----------
    dicom_path : str
        Path to a single DICOM file or a folder containig one or more DICOM files
    savedir : str
        Foldername where to store denoised DICOM file(s) in
    method : Union[ Literal[ Methods.RESNET, Methods.CNN10, Methods.DUGAN, Methods.QAE, Methods.REDCNN, Methods.TRANSCT, Methods.WGANVGG, Methods.BILATERAL, ], str, ]
        Enum item or string, specifying model to load. See [ldctbench.hub.methods.Methods][] for more info.
    device : Optional[torch.device], optional
        torch.device, optional
    disable_progress : bool, optional
        Disable progress bar (if denoising multiple DICOM files), by default False

    Raises
    ------
    ValueError
        If provided path is neither a DICOM file nor a folder containing at least one DICOM file

    Examples
    --------
    A comprehensive example using this function is provided in [Getting Started][denoise-dicom-using-pretrained-models].
    """
    # Load the model from the model hub
    net = load_model(method=method, device=device)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Get DICOM file paths to process
    files = None
    if os.path.isdir(dicom_path):
        files = [
            os.path.join(dicom_path, file)
            for file in os.listdir(dicom_path)
            if pydicom.misc.is_dicom(os.path.join(dicom_path, file))
        ]
    elif pydicom.misc.is_dicom(dicom_path):
        files = [dicom_path]
    if not files or len(files) == 0:
        raise ValueError(f"Couldn't find any DICOM file at {dicom_path}!")

    # Iterate over filepaths
    for file in tqdm(files, desc="Denoise DICOMs", disable=disable_progress):
        ds = pydicom.read_file(file)

        # Training data had RescaleIntercept: -1024 HU and RescaleSlope: 1.0. raise a warning if given dicom has different values
        intercept = getattr(ds, "RescaleIntercept", None)
        slope = getattr(ds, "RescaleSlope", None)
        if not intercept or int(intercept) != -1024:
            warnings.warn(
                f"Expected DICOM data to have intercept -1024 but got {intercept} instead!"
            )
        if not intercept or float(slope) != 1.0:
            warnings.warn(
                f"Expected DICOM data to have slope 1.0 but got {slope} instead!"
            )

        # Might be that this function doesn't properly handle DICOM files with compressed pixel data. Raise a warning.
        if not ds.file_meta.TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian:
            ds.decompress()
            warnings.warn(
                f"TransferSyntaxUID (0002, 0010) must be {pydicom.uid.ExplicitVRLittleEndian} ({pydicom.uid.ExplicitVRLittleEndian.name}) but got {ds.file_meta.TransferSyntaxUID} ({ds.file_meta.TransferSyntaxUID.name}) instead! Used pydicoms .decompress()"
            )

        # Denoise image data
        x = ds.pixel_array
        x_denoised = denoise_numpy(
            x.astype("float32"), method=net, device=device, num_slices_parallel=1
        )

        # Clip to supported data range
        np.clip(
            x_denoised,
            a_min=np.iinfo(x.dtype).min,
            a_max=np.iinfo(x.dtype).max,
            out=x_denoised,
        )

        # Overwrite existing PixelData
        ds.PixelData = x_denoised.astype(x.dtype).tobytes()

        # Save as a new file
        new_filepath = os.path.join(savedir, os.path.basename(file))
        if os.path.exists(new_filepath):
            warnings.warn(
                f"Adding suffix {method.value} to filename as {new_filepath} already exists!"
            )
            filename_split = os.path.basename(file).split(".")
            new_filepath = os.path.join(
                savedir,
                ".".join(filename_split[:-1]) + f"_{method.value}.{filename_split[-1]}",
            )
        ds.save_as(new_filepath)
