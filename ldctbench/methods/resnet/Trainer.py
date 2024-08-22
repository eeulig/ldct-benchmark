from argparse import Namespace

import torch
import torch.nn as nn
import utils
from methods.base import BaseTrainer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer class for LDCT ResNet[^1]

    [^1]: A. D. Missert, S. Leng, L. Yu, and C. H. McCollough, “Noise subtraction for low-dose CT images using a deep convolutional neural network,” in Proceedings of the Fifth International Conference on Image Formation in X-Ray Computed Tomography, Salt Lake City, UT, USA, May 2018, pp. 399–402.
    """

    def __init__(self, args: Namespace, device: torch.device):
        """Init function

        Parameters
        ----------
        args : Namespace
            Arguments to configure the trainer.
        device : torch.device
            Torch device to use for training.
        """
        super().__init__(args, device)
        self.criterion = nn.MSELoss()
        self.model = Model(args).to(self.dev)
        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)
        self.optimizer = utils.setup_optimizer(args, self.model.parameters())
