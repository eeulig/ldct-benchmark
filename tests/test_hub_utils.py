import os
import shutil
import tempfile
import warnings

import numpy as np
import pydicom
import pytest
import torch

from ldctbench.hub import Methods
from ldctbench.hub.utils import denoise_dicom, denoise_numpy

warnings.filterwarnings("error")


class DicomData:
    def __init__(self, file_id):
        self.file_id = file_id
        self.tempdir = tempfile.TemporaryDirectory()
        self.download_dicom()
        self.folder_path = self.tempdir.name
        self.dicom_path = os.path.join(self.folder_path, f"{self.file_id}.dcm")

    def download_dicom(self):
        os.system(
            f's5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp "s3://idc-open-data/b9cf8e7a-2505-4137-9ae3-f8d0cf756c13/{self.file_id}.dcm" {self.tempdir.name}'
        )

    def cleanup(self):
        for item in os.listdir(self.folder_path):
            if os.path.join(self.folder_path, item) == self.dicom_path:
                continue
            if os.path.isfile(os.path.join(self.folder_path, item)):
                os.remove(os.path.join(self.folder_path, item))
            else:
                shutil.rmtree(os.path.join(self.folder_path, item))

    def destroy(self):
        self.tempdir.cleanup()


@pytest.fixture(scope="session")
def sample_dicom():
    dicom_data = DicomData("496788de-f0f0-41fd-b19a-6da82268fd0a")
    yield dicom_data
    print(dicom_data.folder_path)
    dicom_data.destroy()


def test_denoise_random_2D_numpy_array_normalized():
    x = np.random.normal(size=(64, 64))  # Make it fast for testing
    y = denoise_numpy(
        x=x, method=Methods.CNN10, device=torch.device("cpu"), is_normalized=True
    )
    assert x.shape == y.shape


def test_denoise_random_2D_numpy_array_not_normalized():
    x = np.random.normal(
        loc=481.0, scale=502.0, size=(64, 64)
    )  # Make it fast for testing
    y = denoise_numpy(x=x, method=Methods.CNN10, device=torch.device("cpu"))
    assert x.shape == y.shape


def test_denoise_random_3D_numpy_array_normalized():
    x = np.random.normal(size=(1, 64, 64))  # Make it fast for testing
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


def test_denoise_random_2D_numpy_array_no_device_given():
    x = np.random.normal(size=(64, 64))  # Make it fast for testing
    y = denoise_numpy(x=x, method=Methods.CNN10)
    assert x.shape == y.shape


def test_denoise_numpy_when_given_1D_array_raise_error():
    x = np.random.normal(size=(64))  # Make it fast for testing
    with pytest.raises(ValueError):
        denoise_numpy(x=x, method=Methods.CNN10, device=torch.device("cpu"))


def test_denoise_numpy_when_given_4D_array_raise_error():
    x = np.random.normal(size=(2, 5, 64, 64))  # Make it fast for testing
    with pytest.raises(ValueError):
        denoise_numpy(x=x, method=Methods.CNN10, device=torch.device("cpu"))


def test_denoise_dicom_single_file_when_savedir_is_same_as_dicom_path_raise_warning(
    sample_dicom,
):
    # Apply network and store to same folder
    with pytest.warns() as record:
        denoise_dicom(
            dicom_path=sample_dicom.dicom_path,
            savedir=sample_dicom.folder_path,
            method=Methods.CNN10,
        )

    # Check that we raised a duplicate file warning
    assert len(record) == 1
    assert str(record[0].message).startswith("Adding suffix")

    # Check that dicoms are identical except for PixelData DICOM tag
    ds1 = pydicom.read_file(sample_dicom.dicom_path)
    ds2 = pydicom.read_file(
        os.path.join(
            sample_dicom.folder_path,
            f"{sample_dicom.file_id}_{Methods.CNN10.value}.dcm",
        )
    )
    diffs = [
        (elem1.tag.group, elem1.tag.element)
        for (elem1, elem2) in zip(ds1, ds2)
        if elem1 != elem2
    ]

    assert len(diffs) == 1
    assert diffs[0] == (32736, 16)

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_folder_when_savedir_is_same_as_dicom_path_raise_warning(
    sample_dicom,
):
    # Apply network and store to same folder
    with pytest.warns() as record:
        denoise_dicom(
            dicom_path=sample_dicom.folder_path,
            savedir=sample_dicom.folder_path,  # Store to same folder (will raise warning)
            method=Methods.CNN10,
        )
    # Check that we raised a duplicate file warning
    assert len(record) == 1
    assert str(record[0].message).startswith("Adding suffix")

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_folder_when_savedir_is_different_from_dicom_path_raise_no_warning(
    sample_dicom,
):
    # Apply network and store to different folder
    with warnings.catch_warnings():
        denoise_dicom(
            dicom_path=sample_dicom.folder_path,
            savedir=os.path.join(sample_dicom.folder_path, "denoised"),
            method=Methods.CNN10,
        )

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_when_given_empty_folder_raise_error():
    empty_tmpdir = tempfile.TemporaryDirectory()
    with pytest.raises(ValueError):
        denoise_dicom(
            dicom_path=empty_tmpdir.name,
            savedir=empty_tmpdir.name,
            method=Methods.CNN10,
        )
    empty_tmpdir.cleanup()


def test_denoise_dicom_when_wrong_intercept_raises_warning(sample_dicom):
    # Copy DICOM and manipulate intercept
    manipulated_dicom_path = os.path.join(
        sample_dicom.folder_path, f"{sample_dicom.file_id}_wrong_intercept.dcm"
    )
    ds = pydicom.read_file(sample_dicom.dicom_path)
    ds.RescaleIntercept = "0"
    ds.save_as(manipulated_dicom_path)

    # Apply network
    with pytest.warns() as record:
        denoise_dicom(
            dicom_path=manipulated_dicom_path,
            savedir=os.path.join(sample_dicom.folder_path, "denoised"),
            method=Methods.CNN10,
        )

    # Check that we raised a warning about the wrong intercept
    assert len(record) == 1
    assert str(record[0].message).startswith("Expected DICOM data to have intercept")

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_when_wrong_slope_raises_warning(sample_dicom):
    # Copy DICOM and manipulate intercept
    manipulated_dicom_path = os.path.join(
        sample_dicom.folder_path, f"{sample_dicom.file_id}_wrong_slope.dcm"
    )
    ds = pydicom.read_file(sample_dicom.dicom_path)
    ds.RescaleSlope = "0"
    ds.save_as(manipulated_dicom_path)

    # Apply network
    with pytest.warns() as record:
        denoise_dicom(
            dicom_path=manipulated_dicom_path,
            savedir=os.path.join(sample_dicom.folder_path, "denoised"),
            method=Methods.CNN10,
        )

    # Check that we raised a warning about the wrong intercept
    assert len(record) == 1
    assert str(record[0].message).startswith("Expected DICOM data to have slope")

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_when_dicom_pixel_data_is_compressed(sample_dicom):
    # Copy DICOM and compress pixel data
    compressed_dicom_path = os.path.join(
        sample_dicom.folder_path, f"{sample_dicom.file_id}_compressed.dcm"
    )
    ds = pydicom.read_file(sample_dicom.dicom_path)
    ds.compress(transfer_syntax_uid=pydicom.uid.RLELossless)
    ds.save_as(compressed_dicom_path)

    # Apply network
    denoise_dicom(
        dicom_path=compressed_dicom_path,
        savedir=os.path.join(sample_dicom.folder_path, "denoised"),
        method=Methods.CNN10,
    )

    # Check that dicoms are identical except for PixelData DICOM tag
    ds1 = pydicom.read_file(compressed_dicom_path)
    ds2 = pydicom.read_file(sample_dicom.dicom_path)

    diffs = [
        (elem1.tag.group, elem1.tag.element)
        for (elem1, elem2) in zip(ds1, ds2)
        if elem1 != elem2
    ]  # Iterate over regular DICOM tags
    diffs += [
        (elem1.tag.group, elem1.tag.element)
        for (elem1, elem2) in zip(ds1.file_meta, ds2.file_meta)
        if elem1 != elem2
    ]  # Iterate over file meta tags (this is where TransferSyntaxUID is stored)

    assert len(diffs) == 2
    assert (32736, 16) in diffs
    assert (2, 16) in diffs

    # Delete all files except the original DICOM
    sample_dicom.cleanup()


def test_denoise_dicom_when_unsupported_transfer_syntax_raises_error(sample_dicom):
    # Copy DICOM and manipulate transfer syntax
    manipulated_dicom_path = os.path.join(
        sample_dicom.folder_path,
        f"{sample_dicom.file_id}_unsupported_transfer_syntax.dcm",
    )
    ds = pydicom.read_file(sample_dicom.dicom_path)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.DeflatedExplicitVRLittleEndian
    ds.save_as(manipulated_dicom_path)

    # Apply network. This should raise an error
    with pytest.raises(ValueError):
        denoise_dicom(
            dicom_path=manipulated_dicom_path,
            savedir=os.path.join(sample_dicom.folder_path, "denoised"),
            method=Methods.CNN10,
        )

    # Delete all files except the original DICOM
    sample_dicom.cleanup()
