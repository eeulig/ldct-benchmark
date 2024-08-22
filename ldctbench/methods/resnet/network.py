from argparse import Namespace

import torch.nn as nn


class ResBlock(nn.Module):
    """Single Residual block

    Each block consists of Conv->BN->ReLU->GroupConv->BN->ReLU->Conv +
                            |________________________________________|
    """

    def __init__(self, ch: int):
        """Init function

        Parameters
        ----------
        ch : int
            Number of features to use for each conv layer.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, groups=8),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return self.layers(x) + x


class Model(nn.Module):
    """Residual network

    As proposed in the paper, the network performs noise subtraction on the input.
    """

    def __init__(self, args: Namespace, n_channels: int = 128, n_blocks: int = 10):
        """Init function

        Parameters
        ----------
        args : Namespace
            Command line arguments passed to the model.
        n_channels : int, optional
            Number of features for each conv layer, by default 128
        n_blocks : int, optional
            Number of residual blocks, by default 10
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.in_conv = nn.Conv2d(1, n_channels, 9, padding=4)
        self.out_conv = nn.Conv2d(n_channels, 1, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(n_channels) for _ in range(n_blocks)])

    def forward(self, x):
        res = x
        x = self.in_conv(x)
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.out_conv(x)
        return res - x
