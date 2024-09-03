import os
import tempfile

import numpy as np
import pydicom
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
    file_id = "496788de-f0f0-41fd-b19a-6da82268fd0a"
    os.system(
        f's5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp "s3://idc-open-data/b9cf8e7a-2505-4137-9ae3-f8d0cf756c13/{file_id}.dcm" {tempdir.name}'
    )

    # Apply network
    denoise_dicom(
        dicom_path=tempdir.name,
        savedir=tempdir.name,  # Store to same folder (will raise warning)
        method=Methods.CNN10,
        device=torch.device("cpu"),
    )

    # Test that dicoms are identical except for PixelData DICOM tag
    ds1 = pydicom.read_file(os.path.join(tempdir.name, f"{file_id}.dcm"))
    ds2 = pydicom.read_file(
        os.path.join(tempdir.name, f"{file_id}_{Methods.CNN10.value}.dcm")
    )
    diffs = [
        (elem1.tag.group, elem1.tag.element)
        for (elem1, elem2) in zip(ds1, ds2)
        if elem1 != elem2
    ]

    assert len(diffs) == 1
    assert diffs[0] == (32736, 16)

    # Cleanup tempdir
    tempdir.cleanup()
