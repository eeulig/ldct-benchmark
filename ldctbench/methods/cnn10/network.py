from argparse import Namespace

import torch.nn as nn


class Model(nn.Module):
    """CNN-10 network.
    As described in the original paper, we use a simple 3 layer CNN with following parameters

        Layer 1: Kernel size 9x9, number of filters 64
        Layer 2: Kernel size 3x3, number of filters 32
        Layer 3: Kernel size 5x5, number of filters 1.

    They use ReLU as nonlinearity and no BatchNorm.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 64, 9, padding=4),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 1, 5, padding=2),
            ]
        )

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x
