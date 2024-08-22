from argparse import Namespace
from typing import List

import torch
import torch.nn as nn


class Model(nn.Module):
    """Generator for WGAN-VGG"""

    def __init__(self, args: Namespace):
        """Init function

        Parameters
        ----------
        args : Namespace
            Command line arguments passed to the model.
        """
        super().__init__()
        layers = [nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32, 32, 3, 1, 1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32, 1, 3, 1, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class Discriminator(nn.Module):
    """Discriminator (Critic) for WGAN-VGG training"""

    def __init__(self, input_size: int):
        """Init function

        Parameters
        ----------
        input_size : int
            Input size of images fed in forward pass.
        """
        super().__init__()

        layers = []
        ch_stride_set = [
            (1, 64, 1),
            (64, 64, 2),
            (64, 128, 1),
            (128, 128, 2),
            (128, 256, 1),
            (256, 256, 2),
        ]
        for ch_in, ch_out, stride in ch_stride_set:
            self.add_block(layers, ch_in, ch_out, stride)

        self.output_size = self.conv_output_size(input_size, [3] * 6, [1, 2] * 3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256 * self.output_size * self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    @staticmethod
    def conv_output_size(
        input_size: int, kernel_size_list: List[int], stride_list: List[int]
    ) -> int:
        """Compute output size after feature extractor.

        Parameters
        ----------
        input_size : int
            Input size of images fed in forward pass.
        kernel_size_list : List[int]
            List of kernel sizes for each layer.
        stride_list : List[int]
            List of strides for each layer.

        Returns
        -------
        int
            Output size after feature extractor.
        """
        n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
        for k, s in zip(kernel_size_list[1:], stride_list[1:]):
            n = (n - k) // s + 1
        return n

    @staticmethod
    def add_block(
        layers: List[nn.Module], ch_in: int, ch_out: int, stride: int
    ) -> List[nn.Module]:
        """Append Conv -> LeakyReLU block to layer list.

        Parameters
        ----------
        layers : List[nn.Module]
            List of layers
        ch_in : int
            Number of input features
        ch_out : int
            Number of output features
        stride : int
            Desired stride of the conv layer

        Returns
        -------
        List[nn.Module]
            Layer list with appended layer.
        """
        layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
        layers.append(nn.LeakyReLU())
        return layers

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256 * self.output_size * self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out
