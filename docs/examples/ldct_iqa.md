!!! info "Prerequisite"
    This example assumes you have:

    1. the package `ldct-benchmark` installed
    2. the LDCT dataset downloaded to a folder `path/to/ldct-data`
    3. The environment variable `LDCTBENCH_DATAFOLDER` set to that folder
    
    Please refer to [Getting Started](../getting_started.md) for instructions on how to do these steps.

Besides the standard full-reference IQA metrics (SSIM, PSNR, VIF), we also support evaluation using a specialized model for no-reference perceptual image quality assessment of low-dose CT images. This model was the winning entry (agaldran[^1]) of the *Low-dose Computed Tomography Perceptual Image Quality Assessment Grand Challenge 2023*[^2] which was organized in conjunction with MICCAI 2023.

The aim of the challenge was to develop no-reference IQA methods that correlate well with scores provided by radiologists. To this end, the organizers generated a total of 1500 (1000 train, 200 val, 300 test) images of various quality by introducing noise and streak artifacts into routine-dose abdominal CT images. Resulting images were rated by radiologists on a five-point Likert scale and their mean score was used as the ground truth.

The five-point Likert scale was defined as follows (see Table 1 in the paper[^2]):
    
| Numeric score | Verbal descriptive scale | Diagnostic quality criteria                             | 
|---------------|--------------------------|---------------------------------------------------------|
| 0             | Bad                      | Desired features are not shown                          |
| 1             | Poor                     | Diagnostic interpretation is impossible                 |
| 2             | Fair                     | Images are suitable for limited clinical interpretation |
| 3             | Good                     | Images are suitable for diagnostic interpretation       |
| 4             | Excellent                | The anatomical structure is evident                     |

Given some CT image, the model predicts a score in the range `[0, 4]` (with increments of 0.2) on above scale.

[^1]: Official GitHub repository of agaldran: <https://github.com/agaldran/ldct_iqa>.

[^2]: Lee, Wonkyeong, Fabian Wagner, Adrian Galdran, Yongyi Shi, Wenjun Xia, Ge Wang, Xuanqin Mou, et al. 2025. “Low-Dose Computed Tomography Perceptual Image Quality Assessment.” Medical Image Analysis 99 (January):103343. <https://doi.org/10.1016/j.media.2024.103343>.


!!! warning "Use with out-of-distribution (OOD) data"
    Be aware that any evaluation using this model will most likely be an OOD setting and predicted scores should be interpreted with caution. The model was
    
    - trained only using abodminal CT images. However, the paper[^2] performs some experiments using a clinical head CT dataset, to evaluate the methods generalization capabilities.
    - trained on four distinct noise levels only. These noise levels may not be representative of your data.
    - not trained on denoised images at all. It has only seen routine-dose images and various distorted versions thereof. It remains unclear how well the model generalizes to denoised images.


## Evaluate DICOMs using the LDCTIQA model
Let's evaluate a routine-dose and low-dose abdominal CT scan from the LDCT dataset using this LDCTIQA model (see [ldctbench.evaluate.ldct_iqa.LDCTIQA][] for more details).

```python
# Import libraries
import os
from importlib.resources import files

import numpy as np
import pydicom

from ldctbench.evaluate import LDCTIQA
from ldctbench.utils import load_yaml

# Get abdominal patient from test set
info = load_yaml(files("ldctbench.data").joinpath("info.yml"))
patient = info["test_set"][5]

# Setup model
ldctiqa = LDCTIQA()

# Evaluation function
def evaluate_dicom(folder):
    res = []
    for f in os.listdir(folder):
        file = os.path.join(folder, f)
        if pydicom.misc.is_dicom(file):
            ds = pydicom.filereader.dcmread(file)
            img = ds.pixel_array.astype("float32")
            score = ldctiqa(img)
            res.append(score.item())
    return res

# Evaluate on low-dose and high-dose images
for scan, dose in [("input", "low"), ("target", "high")]:
    scores = evaluate_dicom(
        folder=os.path.join(os.environ["LDCTBENCH_DATAFOLDER"], patient[scan]),
    )
    print(
        f"LDCTIQA for {dose}-dose scan of {patient['id']}: {np.mean(scores):.2f} ± {np.std(scores):.2f}"
    )
```
Running this script will give us the following scores for the routine-dose and low-dose images of patient `L241`:

| Dose level | LDCTIQA score |
|------------|---------------|
| Low        | 2.55 ± 0.44   |
| High       | 4.0 ± 0       |

According to our model, this low-dose scan has an image quality between "Fair" and "Good", while the high-dose scan is rated as "Excellent".

We also provide a function ([ldctbench.evaluate.evaluate_dicom][]) to evaluate a single DICOM file or a folder of DICOM files and store the results in a `json` file (if the `savedir` argument is provided). This function can be used as follows:

```python
import numpy as np
from ldctbench.evaluate import evaluate_dicom
scores = evaluate_dicom(dicom_path="path/to/dicom/series", savedir="path/to/save")
print(f"Evaluated {len(scores)} files. LDCTIQA = {np.mean(scores):.2f} ± {np.std(scores):.2f}")
```
```
>>> Evaluate DICOM(s): 100%|██████████████████| 137/137 [00:11<00:00, 12.27it/s]
>>> Saved scores to path/to/save/scores.json
>>> Evaluated 137 files. LDCTIQA = 3.96 ± 0.08
```

## Evaluate denoising methods using the LDCTIQA model
We can also evaluate the performance of denoising methods using the LDCTIQA model. Let's evaluate the performance of the CNN-10 model on the test set of the LDCT dataset.

Running
```shell
ldctbench-test --methods cnn10 --metrics LDCTIQA --print_table
```
will give us the following results:

| Method   |   LDCTIQA (Chest) |   LDCTIQA (Abdomen) |   LDCTIQA (Neuro) |
|----------|-------------------|---------------------|-------------------|
| LD       |             0     |               3.208 |             3.965 |
| cnn10    |             3.296 |               3.998 |             3.972 |

We find that on all three anatomies, the CNN-10 model improves over the LD baseline in terms of LDCTIQA score and provides a rating between "Good" and "Excellent" for all anatomies. However, for the neuro scans the improvement is marginal. **Note that results on chest and neuro scans should be interpreted with caution, as the model was trained on abdominal CT images only!**