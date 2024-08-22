import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm


class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
        super(QuadConv, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.padding = padding
        self.valid_padding = ["valid", "same"]
        if self.padding not in self.valid_padding:
            raise ValueError(
                f"Padding must be one of {self.valid_padding}. Got {self.padding} instead!"
            )

        self.W_r = nn.Parameter(
            torch.Tensor(
                truncnorm.rvs(
                    -2, 2, scale=0.1, size=[out_channels, in_channels, *kernel_size]
                )
            )
        )
        self.W_g = nn.Parameter(
            torch.zeros(
                size=[out_channels, in_channels, *kernel_size], dtype=torch.float32
            )
        )
        self.W_b = nn.Parameter(
            torch.zeros(
                size=[out_channels, in_channels, *kernel_size], dtype=torch.float32
            )
        )
        self.b_r = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))
        self.b_g = nn.Parameter(torch.ones(size=[out_channels], dtype=torch.float32))
        self.b_b = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))

    def forward(self, x):
        x1 = F.conv2d(x, self.W_r, self.b_r, stride=1, padding=self.padding)
        x2 = F.conv2d(x, self.W_g, self.b_g, stride=1, padding=self.padding)
        x3 = F.conv2d(x * x, self.W_b, self.b_b, stride=1, padding=self.padding)
        return x1 * x2 + x3


class QuadDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
        super(QuadDeconv, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        self.padding = padding
        self.valid_padding = ["valid", "same"]
        if self.padding not in self.valid_padding:
            raise ValueError(
                f"Padding must be one of {self.valid_padding}. Got {self.padding} instead!"
            )

        self.W_r = nn.Parameter(
            torch.Tensor(
                truncnorm.rvs(
                    -2,
                    2,
                    scale=0.1,
                    size=[in_channels, out_channels, *self.kernel_size],
                )
            )
        )
        self.W_g = nn.Parameter(
            torch.zeros(
                size=[in_channels, out_channels, *self.kernel_size], dtype=torch.float32
            )
        )
        self.W_b = nn.Parameter(
            torch.zeros(
                size=[in_channels, out_channels, *self.kernel_size], dtype=torch.float32
            )
        )
        self.b_r = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))
        self.b_g = nn.Parameter(torch.ones(size=[out_channels], dtype=torch.float32))
        self.b_b = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))

    def forward(self, x):
        pad = (
            int(np.ceil((self.kernel_size[0] - 1) / 2)) if self.padding == "same" else 0
        )
        x1 = F.conv_transpose2d(x, self.W_r, self.b_r, stride=1, padding=(pad, pad))
        x2 = F.conv_transpose2d(x, self.W_g, self.b_g, stride=1, padding=(pad, pad))
        x3 = F.conv_transpose2d(x * x, self.W_b, self.b_b, padding=(pad, pad))
        return x1 * x2 + x3


class Model(nn.Module):
    r"""Quadratic Autoencoder (QAE)

    In the paper, the authors descibe their initialization as follows:
    As far as the Q-AE is concerned, parameters $w_r$ and $w_g$ of each layer were randomly initialized with a truncated Gaussian
    function, $b_g$ are set to 1 for all the layers. In this way, quadratic term $(w_r x^T + b_r )(w_g x^T + b_g)$ turns into linear
    term $(w_r x^T + b_r )$. The reason why we use such initialization is because quadratic terms should not be pre-determined,
    they should be learned in the training. br and c were set to 0 initially for all the layers. wb was set to 0 here, we
    will discuss the influence of wb on the network in the context of direct initialization and transfer learning later.

    In our experiments, the network diverges with these settings. In their [source code](https://github.com/FengleiFan/QAE/blob/e190bde3c18a3e0e319f68787634b7806b77d9d7/Quadratic-Autoencoder.py#L61-L95) the authors also commented out the lines where they initialize $W_g$ as truncated normal, and initialize it with zeros instead. We here follow their official GitHub
    implementation and initialize $W_g$ as zero.
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder = nn.ModuleList(
            [
                QuadConv(1, 15, 3, "same"),
                QuadConv(15, 15, 3, "same"),
                QuadConv(15, 15, 3, "same"),
                QuadConv(15, 15, 3, "same"),
                QuadConv(15, 15, 3, "valid"),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                QuadDeconv(15, 15, 3, "valid"),
                QuadDeconv(15, 15, 3, "same"),
                QuadDeconv(15, 15, 3, "same"),
                QuadDeconv(15, 15, 3, "same"),
                QuadDeconv(15, 1, 3, "same"),
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.encoder[0](x))
        x2 = self.relu(self.encoder[1](x1))
        x3 = self.relu(self.encoder[2](x2))
        x4 = self.relu(self.encoder[3](x3))
        x5 = self.relu(self.encoder[4](x4))
        x6 = self.relu(self.decoder[0](x5) + x4)
        x7 = self.relu(self.decoder[1](x6))
        x8 = self.relu(self.decoder[2](x7) + x2)
        x9 = self.relu(self.decoder[3](x8))
        return self.decoder[4](x9) + x
