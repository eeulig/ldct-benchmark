import os
import tempfile

import numpy as np
import pydicom
import requests
import torch

from ldctbench.hub import Methods
from ldctbench.hub.utils import denoise_dicom, denoise_numpy


def test_denoise_random_2D_numpy_array_normalized():
    x = np.random.normal(size=(64, 64))  # Make it a fast for testing
    y = denoise_numpy(
        x=x, method=Methods.CNN10, device=torch.device("cpu"), is_normalized=True
    )
    assert x.shape == y.shape


def test_denoise_random_2D_numpy_array_not_normalized():
    x = np.random.normal(
        loc=481.0, scale=502.0, size=(64, 64)
    )  # Make it a fast for testing
    y = denoise_numpy(x=x, method=Methods.CNN10, device=torch.device("cpu"))
    assert x.shape == y.shape


def test_denoise_random_3D_numpy_array_normalized():
    x = np.random.normal(size=(1, 64, 64))  # Make it a fast for testing
    y = denoise_numpy(
        x=x, method=Methods.CNN10, device=torch.device("cpu"), is_normalized=True
    )
    assert x.shape == y.shape


def test_denoise_random_3D_numpy_array_not_normalized():
    x = np.random.normal(
        loc=481.0, scale=502.0, size=(1, 64, 64)
    )  # Make it a fast for testing
    y = denoise_numpy(x=x, method=Methods.CNN10, device=torch.device("cpu"))
    assert x.shape == y.shape


def test_denoise_dicom():
    # Download a single DICOM to tempdir
    tempdir = tempfile.TemporaryDirectory()
    r = requests.get("https://dataverse.harvard.edu/api/access/datafile/7576771")
    with open(os.path.join(tempdir.name, "0.dcm"), "wb") as f:
        f.write(r.content)

    # Apply network
    denoise_dicom(
        dicom_path=tempdir.name,
        savedir=tempdir.name,  # Store to same folder (will raise warning)
        method=Methods.CNN10,
        device=torch.device("cpu"),
    )

    # Test that dicoms are identical except for PixelData DICOM tag
    ds1 = pydicom.read_file(os.path.join(tempdir.name, "0.dcm"))
    ds2 = pydicom.read_file(os.path.join(tempdir.name, f"0_{Methods.CNN10.value}.dcm"))
    diffs = [
        (elem1.tag.group, elem1.tag.element)
        for (elem1, elem2) in zip(ds1, ds2)
        if elem1 != elem2
    ]

    assert len(diffs) == 1
    assert diffs[0] == (32736, 16)

    # Cleanup tempdir
    tempdir.cleanup()
