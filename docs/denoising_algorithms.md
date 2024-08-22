## Implemented algorithms
Below we list all methods currently implemented in our benchmark suite

| Name                       | Method name | Paper                                                                                                                                                                                                                                                                                              |
|----------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CNN-10                     | cnn10       | H. Chen, Y. Zhang, W. Zhang, P. Liao, K. Li, J. Zhou, and G. Wang, "Low-dose CT via convolutional neural network,” Biomedical Optics Express, vol. 8, no. 2, pp. 679–694, Jan. 2017                                                                                                                |
| RED-CNN                    | redcnn      | H. Chen, Y. Zhang, M. K. Kalra, F. Lin, Y. Chen, P. Liao, J. Zhou, and G. Wang, “Low-dose CT with a residual encoder-decoder convolutional neural network,” IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524–2535, Dec. 2017                                                        |
| WGAN-VGG                   | wganvgg     | Q. Yang, P. Yan, Y. Zhang, H. Yu, Y. Shi, X. Mou, M. K. Kalra, Y. Zhang, L. Sun, and G. Wang, “Low-dose CT image denoising using a generative adversarial network with wasserstein distance and perceptual loss,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1348– 1357, Jun. 2018. |
| ResNet                     | resnet      | A. D. Missert, S. Leng, L. Yu, and C. H. McCollough, “Noise subtraction for low-dose CT images using a deep convolutional neural network,” in Proceedings of the Fifth International Conference on Image Formation in X-Ray Computed Tomography, Salt Lake City, UT, USA, May 2018, pp. 399–402.   |
| QAE                        | qae         | F. Fan, H. Shan, M. K. Kalra, R. Singh, G. Qian, M. Getzin, Y. Teng, J. Hahn, and G. Wang, “Quadratic autoencoder (Q-AE) for low-dose CT denoising,” IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 2035–2050, Jun. 2020.                                                               |
| DU-GAN                     | dugan       | Z. Huang, J. Zhang, Y. Zhang, and H. Shan, “DU-GAN: Generative adversarial networks with dual-domain U-Net-based discriminators for low-dose CT denoising,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–12, 2022.                                                         |
| TransCT                    | transct     | Z. Zhang, L. Yu, X. Liang, W. Zhao, and L. Xing, “TransCT: Dual-path transformer for low dose computed tomography,” in MICCAI, 2021                                                                                                                                                                |
| Trainable bilateral filter | bilateral   | F. Wagner, M. Thies, M. Gu, Y. Huang, S. Pechmann, M. Patwari, S. Ploner, O. Aust, S. Uderhardt, G. Schett, S. Christiansen, and A. Maier, “Ultralow-parameter denoising: Trainable bilateral filter layers in computed tomography,” Medical Physics, vol. 49, no. 8, pp. 5107– 5120, 2022.        |


## Test set performance
Below we report the results of the best performing networks of each method on the test dataset. They can be reproduced by running `python test.py --print_table` (see [Test models](examples/test_models.md)).

| Method    |   SSIM (Chest) |   SSIM (Abdomen) |   SSIM (Neuro) |   PSNR (Chest) |   PSNR (Abdomen) |   PSNR (Neuro) |   VIF (Chest) |   VIF (Abdomen) |   VIF (Neuro) |
|-----------|----------------|------------------|----------------|----------------|------------------|----------------|---------------|-----------------|---------------|
| LD        |          0.312 |            0.856 |          0.914 |         18.066 |           29.117 |         30.923 |         0.083 |           0.353 |         0.578 |
| cnn10     |          0.559 |            0.907 |          0.928 |         27.307 |           32.737 |         31.968 |         0.175 |           0.455 |         0.642 |
| redcnn    |          0.584 |            0.913 |          0.932 |         28.002 |           33.685 |         34.132 |         0.205 |           0.504 |         0.715 |
| qae       |          0.557 |            0.903 |          0.928 |         27.115 |           32.304 |         31.923 |         0.167 |           0.424 |         0.618 |
| wganvgg   |          0.505 |            0.893 |          0.92  |         25.324 |           30.906 |         29.208 |         0.137 |           0.39  |         0.566 |
| resnet    |          0.581 |            0.912 |          0.932 |         28.032 |           33.583 |         33.853 |         0.21  |           0.5   |         0.705 |
| qae       |          0.557 |            0.903 |          0.928 |         27.115 |           32.304 |         31.923 |         0.167 |           0.424 |         0.618 |
| dugan     |          0.544 |            0.904 |          0.93  |         26.316 |           32.468 |         32.078 |         0.156 |           0.441 |         0.656 |
| transct   |          0.538 |            0.89  |          0.893 |         26.736 |           30.924 |         27.363 |         0.155 |           0.387 |         0.461 |
| bilateral |          0.529 |            0.871 |          0.905 |         25.057 |           27.357 |         29.238 |         0.143 |           0.373 |         0.541 |
