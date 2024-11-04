import os
import tempfile
import warnings

import numpy as np
import pydicom
import pytest
import torch
from PIL import Image
from skimage import img_as_ubyte, io
from torchvision.transforms import Compose, Normalize, ToTensor

from ldctbench.evaluate import LDCTIQA, evaluate_dicom
from ldctbench.utils import load_json

transforms = Compose(
    [ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

IMGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ldct_iqa")


def preprocess_tiff(f_name):
    # Load and preprocess TIFF image
    img = Image.fromarray(
        img_as_ubyte(io.imread(os.path.join(IMGDIR, f_name)))
    ).convert("RGB")
    img = transforms(img).unsqueeze(0)
    return img


def test_ldct_iqa_given_test_images_return_correct_score():
    # Load model
    iqa = LDCTIQA()

    # Check that model returns correct score for given test images
    scores = load_json(IMGDIR + "/scores.json")
    for f_name in scores:
        img = preprocess_tiff(f_name)
        score = torch.tensor(scores[f_name])
        pred = iqa(img.to(iqa.device), preprocess=False)
        assert torch.allclose(pred, score, atol=1e-2)


def test_ldct_iqa_not_given_numpy_raise_error():
    # Load model
    iqa = LDCTIQA()

    # Check that model raises error when not given numpy array
    img = torch.randn(1, 512, 512)
    with pytest.raises(ValueError):
        iqa(img)


def test_ldct_iqa_given_wrong_image_shape_raise_error():
    # Load model
    iqa = LDCTIQA()

    forbidden_shapes = [(128, 128), (1, 1, 128, 128), (1, 3, 128, 128)]
    # Check that model raises error when given wrong image shape
    for shape in forbidden_shapes:
        img = np.random.randn(*shape)
        with pytest.raises(ValueError):
            iqa(img)


def test_ldct_iqa_not_given_tensor_when_preprocessed_raise_error():
    # Load model
    iqa = LDCTIQA()

    # Check that model raises error when not given tensor
    img = np.random.randn(1, 512, 512)
    with pytest.raises(ValueError):
        iqa(img, preprocess=False)


def test_ldct_iqa_given_wrong_tensor_shape_when_preprocessed_raise_error():
    # Load model
    iqa = LDCTIQA()

    forbidden_shapes = [(1, 1, 128, 128), (1, 3, 128, 128), (1, 512, 512), (512, 512)]
    # Check that model raises error when given wrong tensor shape
    for shape in forbidden_shapes:
        img = torch.randn(*shape)
        with pytest.raises(ValueError):
            iqa(img.to(iqa.device), preprocess=False)


def test_evaluate_dicom_file(
    sample_dicom,
):
    # Evaluate single DICOM
    with warnings.catch_warnings():
        scores = evaluate_dicom(
            dicom_path=sample_dicom.dicom_path,
        )

    assert len(scores) == 1


def test_evaluate_dicom_folder(
    sample_dicom,
):
    # Evaluate single DICOM and store json in same folder
    with warnings.catch_warnings():
        scores = evaluate_dicom(
            dicom_path=sample_dicom.folder_path,
            savedir=sample_dicom.folder_path,
        )

    scores_json = load_json(os.path.join(sample_dicom.folder_path, "scores.json"))

    assert len(scores) == 1
    assert scores_json == scores

    # Delete score file
    sample_dicom.cleanup()


def test_evaluate_dicom_when_given_empty_folder_raise_error():
    empty_tmpdir = tempfile.TemporaryDirectory()
    with pytest.raises(ValueError):
        evaluate_dicom(
            dicom_path=empty_tmpdir.name,
        )
    empty_tmpdir.cleanup()


def test_evaluate_dicom_when_wrong_intercept_raises_warning(sample_dicom):
    # Copy DICOM and manipulate intercept
    manipulated_dicom_path = os.path.join(
        sample_dicom.folder_path, f"{sample_dicom.file_id}_wrong_intercept.dcm"
    )
    ds = pydicom.filereader.dcmread(sample_dicom.dicom_path)
    ds.RescaleIntercept = "0"
    ds.save_as(manipulated_dicom_path)

    # Evaluate DICOM
    with pytest.warns() as record:
        evaluate_dicom(
            dicom_path=manipulated_dicom_path,
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
    ds = pydicom.filereader.dcmread(sample_dicom.dicom_path)
    ds.RescaleSlope = "0"
    ds.save_as(manipulated_dicom_path)

    # Evaluate DICOM
    with pytest.warns() as record:
        evaluate_dicom(
            dicom_path=manipulated_dicom_path,
        )

    # Check that we raised a warning about the wrong intercept
    assert len(record) == 1
    assert str(record[0].message).startswith("Expected DICOM data to have slope")

    # Delete all files except the original DICOM
    sample_dicom.cleanup()
