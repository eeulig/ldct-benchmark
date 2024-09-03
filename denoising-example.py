# Make sure that s5cmd is installed! (pip install s5cmd)
import os

import torch

from ldctbench.hub import Methods
from ldctbench.hub.utils import denoise_dicom

# Create folders
folder = "./visible-human"
orig_data = os.path.join(folder, "orig")

if not os.path.exists(orig_data):
    os.makedirs(orig_data)

# Filenames of 10 pelvis slices
files = [
    "496788de-f0f0-41fd-b19a-6da82268fd0a.dcm",
    "a535613b-de28-4080-850a-f5647ee33c96.dcm",
    "9f7ef52e-c93d-430a-9038-970a47e95e3a.dcm",
    "0c7ac013-41e3-404f-9081-9e0cc18f4f67.dcm",
    "2591aad8-7673-4a12-98e0-8984dafa5175.dcm",
    "28494e9b-d274-4310-a0ed-15d4220e1dc1.dcm",
    "f5e41514-d30a-4cef-81ac-fce50b4743d8.dcm",
    "21292f8c-072c-4223-859a-1e70bbc87a42.dcm",
    "5a214c6b-6898-43c1-89f3-52c967dff39e.dcm",
    "cd90f914-2b13-4cd9-9119-976a3c5721c1.dcm",
]

# Download the data
for file in files:
    os.system(
        f's5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp "s3://idc-open-data/b9cf8e7a-2505-4137-9ae3-f8d0cf756c13/{file}" visible-human/orig'
    )

# Apply RED-CNN and DU-GAN and store the processed DICOMs
# to ./visible-human/redcnn and ./visible-human/dugan
for method in [Methods.REDCNN, Methods.DUGAN]:
    denoise_dicom(
        dicom_path=orig_data,
        savedir=os.path.join(folder, method.value),
        method=method,
        device=torch.device(
            "mps"
        ),  # Use "mps" for Apple Silicon, "cuda" for NVIDIA GPUs or "cpu" for CPU
    )
