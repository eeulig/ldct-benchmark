import os
import warnings
from typing import Optional, Union

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torchvision.models import swin_t
from torchvision.transforms import Normalize
from tqdm import tqdm

from ldctbench.hub.load_model import download_checkpoint
from ldctbench.utils.auxiliaries import apply_center_width, save_json

CHECKPOINT = {
    "name": "ldct-iqa",
    "url": "https://downloads.eeulig.com/ldct-benchmark/ldct-iqa.pt",
    "checksum": "4d7596a1638ebb1ccb8945f079b08b30791a1241965f1541dc842d48a0341bc2",
    "kwargs": {"num_classes": 21, "num_heads": 4},
    "center": 1024.0 - 300.0,
    "width": 1400.0,
}


class MultiHeadSwin(torch.nn.Module):
    def __init__(self, num_classes, num_heads):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.model = swin_t(weights=None)
        num_ftrs = self.model.head.in_features
        self.model.head = torch.nn.Identity()

        def get_head(num_ftrs, num_classes):
            return torch.nn.Linear(num_ftrs, num_classes)

        self.heads = torch.nn.ModuleList(
            [get_head(num_ftrs, num_classes) for _ in range(self.num_heads)]
        )

    def forward(self, x):
        x = self.model(x)
        head_predictions = [self.heads[i](x) for i in range(self.num_heads)]
        logits = torch.stack(
            head_predictions, dim=-2
        )  # batch_size x num_heads x num_classes
        s = logits.softmax(dim=-1)  # batch_size x num_heads x num_classes
        out = torch.mean(s, dim=-2)  # batch_size x num_classes
        return out


def load_model(
    checkpoint_dict: dict,
    eval: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:

    if not device:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # Init model
    model = MultiHeadSwin(**checkpoint_dict["kwargs"]).to(device)

    # Download checkpoint
    chkpt_path = download_checkpoint(
        checkpoint_dict["name"],
        checkpoint_dict["url"],
        checksum=checkpoint_dict["checksum"],
    )
    state = torch.load(chkpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if eval:
        model.eval()
    return model


class LDCTIQA:
    """Class to perform no-reference IQA on LDCT images using the winning model of the *Low-dose Computed Tomography Perceptual Image Quality Assessment Grand Challenge 2023*[^1] which was organized in conjunction with MICCAI 2023.

    The aim of the challenge was to develop no-reference IQA methods that correlate well with scores provided by radiologists. To this end, the organizers generated a total of 1,500 (1,000 train, 200 val, 300 test) images of various quality by introducing noise and streak artifacts into routine-dose abdominal CT images. Resulting images were rated by radiologists on a five-point Likert scale and their mean score was used as the ground truth.

    The five-point Likert scale was defined as follows (see Table 1 in the paper[^1]):

    | Numeric score | Verbal descriptive scale | Diagnostic quality criteria                             |
    |---------------|--------------------------|---------------------------------------------------------|
    | 0             | Bad                      | Desired features are not shown                          |
    | 1             | Poor                     | Diagnostic interpretation is impossible                 |
    | 2             | Fair                     | Images are suitable for limited clinical interpretation |
    | 3             | Good                     | Images are suitable for diagnostic interpretation       |
    | 4             | Excellent                | The anatomical structure is evident                     |


    The model that we use here is a slight variation of the winning model (agaldran). The differences to the model used in the challenge are:

    - We retrained including the additional 300 test images from the challenge
    - Only using the muli-head swin transformer (no ResNeXt)
    - No ensemble, only one model on a single training/validation split of the 1,500 images

    !!! warning "Use with out-of-distribution (OOD) data"
        Be aware that any evaluation using this model will most likely be an OOD setting and predicted scores should be interpreted with caution. The model was

        - trained only using abodminal CT images. However, the paper[^1] performs some experiments using a clinical head CT dataset, to evaluate the methods generalization capabilities.
        - trained on four distinct noise levels only. These noise levels may not be representative of your data.
        - not trained on denoised images at all. It has only seen routine-dose images and various distorted versions thereof. It remains unclear how well the model generalizes to denoised images.

    [^1]: Lee, Wonkyeong, Fabian Wagner, Adrian Galdran, Yongyi Shi, Wenjun Xia, Ge Wang, Xuanqin Mou, et al. 2025. “Low-Dose Computed Tomography Perceptual Image Quality Assessment.” Medical Image Analysis 99 (January):103343. <https://doi.org/10.1016/j.media.2024.103343>.

    Parameters
    ----------
    device : torch.device, optional
        Device to run LDCTIQA model on
    """

    def __init__(self, device: Optional[torch.Tensor] = None):
        if not device:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        self.model = load_model(CHECKPOINT, device=self.device)
        self.normalize = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """Preprocess a given CT image

        The function takes a numpy float array in HU + 1024, applies windowing with *(C,W) = (300, 1400)* HU (as was done for the training data) and normalizes the image to the ImageNet mean and standard deviation.

        Parameters
        ----------
        x : np.ndarray
            A `np.ndarray` of shape `[512,512]` or `[1,512,512]` representing a CT image in HU + 1024

        Raises
        ------
        ValueError
            If input is not a numpy array
        ValueError
            If image shape is not `[512,512]` or `[1,512,512]`

        Returns
        -------
        torch.Tensor
            Preprocessed image as torch.Tensor of shape `[1,3,512,512]`
        """

        if not isinstance(x, np.ndarray):
            raise ValueError(f"Expected numpy array but got {type(x)} instead!")

        if x.shape not in [(512, 512), (1, 512, 512)]:
            raise ValueError(
                f"Expected image shape to be (512,512) or (1,512,512) but got {x.shape} instead!"
            )

        x = apply_center_width(
            x, center=CHECKPOINT["center"], width=CHECKPOINT["width"]
        )

        x_t = torch.tensor(x).repeat(1, 3, 1, 1)  # B x C x H x W
        x_t = self.normalize(x_t).to(self.device)
        return x_t

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, np.ndarray], preprocess: bool = True
    ) -> torch.Tensor:
        """Predict the IQA score for a given image

        Parameters
        ----------
        x : Union[torch.Tensor, np.ndarray]
            if `preprocess` is `True`, expects numpy float array in HU + 1024 (i.e., air should have a value of ~24) of shape `[1,512,512]`. Otherwise, expects preprocessed torch tensor of shape `[B,3,512,512]`
        preprocess : bool, optional
            Whether inputs should be preprocessed (i.e., windowed and normalized)

        Returns
        -------
        torch.Tensor
            Predicted score on a five-point Likert scale `[0,4]` in `0.2` increments

        Examples
        --------
        Evaluate IQA on a single DICOM slice

        >>> import pydicom
        >>> ds = pydicom.filereader.dcmread("path/to/dicom")
        >>> img = ds.pixel_array.astype("float32")
        >>> # Ensure that image has expected offset and scaling
        >>> assert ds.RescaleSlope == "1" and ds.RescaleIntercept == "-1024"
        >>> iqa = LDCTIQA()
        >>> score = iqa(img)
        """

        if preprocess:
            x = self.preprocess(x)

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor but got {type(x)} instead!")

        if not (len(x.shape) == 4 and x.shape[1:] == (3, 512, 512)):
            raise ValueError(
                f"Expected image shape to be (B,3,512,512) but got {x.shape} instead!"
            )

        pred = self.model(x)
        score = torch.argmax(pred, -1) / 5.0
        return score


def evaluate_dicom(
    dicom_path: str,
    savedir: Optional[str] = None,
    device: Optional[torch.device] = None,
    disable_progress: bool = False,
) -> list:
    """Evaluate LDCTIQA on a single DICOM file or a series of DICOM files in a folder

    Parameters
    ----------
    dicom_path : str
        Path to a single DICOM file or a folder containig one or more DICOM files
    savedir : str, optional
        Foldername where to store evaluation results. If not provided, results are not saved, only returned
    device : Optional[torch.device], optional
        torch.device, optional
    disable_progress : bool, optional
        Disable progress bar, by default False

    Returns
    -------
    list
        List of IQA scores for each DICOM file

    Raises
    ------
    ValueError
        If provided path is neither a DICOM file nor a folder containing at least one DICOM file
    ValueError
        If image shape is not `512 x 512`

    Examples
    --------
    Evaluate a folder of DICOM files and save the results to a file `scores.json` in the provided `savedir`:
    ```python
    from ldctbench.evaluate import evaluate_dicom
    scores = evaluate_dicom(dicom_path="path/to/dicom/series", savedir="path/to/save")
    ```
    Evalauate a folder of DICOM files and return scores (don't save them):
    ```python
    from ldctbench.evaluate import evaluate_dicom
    scores = evaluate_dicom(dicom_path="path/to/dicom/series")
    ```

    The function is also documented in [this example][evaluate-dicoms-using-the-ldctiqa-model].
    """

    # Setup model
    model = LDCTIQA(device=device)

    if savedir and not os.path.exists(savedir):
        os.makedirs(savedir)

    # Get DICOM file paths to process
    files = None
    if os.path.isdir(dicom_path):
        files = [
            os.path.join(dicom_path, item)
            for item in os.listdir(dicom_path)
            if (
                not os.path.isdir(os.path.join(dicom_path, item))
                and pydicom.misc.is_dicom(os.path.join(dicom_path, item))
            )
        ]
    elif pydicom.misc.is_dicom(dicom_path):
        files = [dicom_path]
    if not files or len(files) == 0:
        raise ValueError(f"Couldn't find any DICOM file at {dicom_path}!")

    scores = []
    # Iterate over filepaths
    for file in tqdm(files, desc="Evaluate DICOM(s)", disable=disable_progress):
        ds = pydicom.filereader.dcmread(file)

        # Training data had RescaleIntercept: -1024 HU and RescaleSlope: 1.0. Raise a warning if given dicom has different values
        intercept = getattr(ds, "RescaleIntercept", None)
        slope = getattr(ds, "RescaleSlope", None)
        if intercept is None or int(intercept) != -1024:
            warnings.warn(
                f"Expected DICOM data to have intercept -1024 but got {intercept} instead!"
            )
        if slope is None or float(slope) != 1.0:
            warnings.warn(
                f"Expected DICOM data to have slope 1.0 but got {slope} instead!"
            )

        # Evaluate DICOM using model
        x = ds.pixel_array.astype("float32")
        score = model(x)
        scores.append(round(score.item(), 2))

    # Save scores to json
    if savedir:
        outfile = os.path.join(savedir, "scores.json")
        save_json(scores, outfile)
        print(f"Saved scores to {outfile}")

    return scores
