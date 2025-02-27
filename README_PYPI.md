![LDCT Benchmark](https://github.com/eeulig/ldct-benchmark/raw/main/docs/assets/header_light.svg)

# Benchmarking Deep Learning-Based Low Dose CT Image Denoising Algorithms
[GitHub](https://github.com/eeulig/ldct-benchmark) | [Documentation](https://eeulig.github.io/ldct-benchmark/)

## Why low dose CT?
Computed tomography (CT) is an important imaging modality, with numerous applications including biology, medicine, and nondestructive testing. However, the use of ionizing radiation remains a key concern and thus clinical CT scans must follow the ALARA (as low as reasonably achievable) principle. Therefore, reducing the dose and thus radiation risk is of utmost importance and one of the primary research areas in the field. A straightforward way to reduce dose is by lowering the tube current (i.e., reducing the X-Ray intensity). However, this comes at the cost of deteriorated image quality due to increased image noise and thus potentially reduced diagnostic value. To alleviate this drawback, numerous algorithms have been proposed to solve the task of low-dose CT (LDCT) denoising, i.e., reducing image noise in the reconstructed image.

## Why this project?
We found that LDCT denoising algorithms proposed in the literature are difficult to compare due to the following reasons:

**Different datasetes:** Most LDCT denoising methods are trained and evaluated on (subsets of) one of the following two datasets

- *2016 NIHAAPMMayo Clinic LDCT Grand Challenge*
- *LDCT and Projection data*

However, authors of each method employ their own training, validation, and test split and thus reported metrics are not comparable across publications.

**Unfair choice of hyperparameters:** Very few publications in the field report the use of hyperparameter optimization. And even if *some* form of hyperparameter optimization is employed, this is usually limited to the newly proposed method. For the comparison methods, authors often use the hyperparameters reported in the reference. However, the optimal hyperparameters for a given method may be dataset-specific, meaning that the parameters tuned by authors $A$ for their dataset $\mathcal{D}_A$ might not generalize to another dataset $\mathcal{D}_B$ used by authors $B$ in their experiments.

**Missing open source implementations:** Many authors don't provide open-source implementations of their algorithms and thus researchers are often left to implement comparison methods themselves. This increases the chance of errors and generally hinders reproducibility.

**Inadequate metrics:** Most LDCT denoising methods are evaluated using SSIM, peak signal-to-noise ratio (PSNR), or root-mean-square error (RMSE). While these are common metrics to quantify performance for natural image denoising, they are usually not in agreement with human readers for medical images.

Therefore, the **aim** of this project is to

1. provide a unified benchmark suite which serves as a reference for existing LDCT denoising algorithms (including optimal hyperparameters) and allows for a fair and reproducible comparison of new methods.
2. make these algorithms easily accessible for practitioners by providing the trained models together with utility functions to denoise CT images.
3. establish novel metrics for the evaluation of LDCT denoising algorithms.

## Documentation
Please read our [documentation](https://eeulig.github.io/ldct-benchmark/) for details on installation and usage of this project.

## Contribute an algorithm
We welcome contributions of novel denoising algorithms. For details on how to do so, please check out our [contributing guide](https://github.com/eeulig/ldct-benchmark/blob/main/CONTRIBUTING.md) or reach out to [me](mailto:elias.eulig@dkfz.de).

## Reference
If you find this project useful for your work, please cite our [Medical Physics paper](https://doi.org/10.1002/mp.17379):
> E. Eulig, B. Ommer, and M. Kachelrieß, “Benchmarking deep learning-based low-dose CT image denoising algorithms,” Medical Physics, vol. 51, no. 12, pp. 8776–8788, Dec. 2024.

```bibtex
@article{ldctbench-medphys,
  title = {Benchmarking deep learning-based low-dose CT image denoising algorithms},
  author = {Eulig, Elias and Ommer, Björn and Kachelrieß, Marc},
  journal = {Medical Physics},
  volume = {51},
  number = {12},
  pages = {8776-8788},
  doi = {https://doi.org/10.1002/mp.17379},
  url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17379},
  year = {2024}
}
```
