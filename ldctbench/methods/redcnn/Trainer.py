from argparse import Namespace

import torch
import torch.nn as nn

from ldctbench.methods.base import BaseTrainer
from ldctbench.utils.training_utils import setup_optimizer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer class for RED-CNN[^1]

    [^1]: H. Chen et al., “Low-dose CT with a residual encoder-decoder convolutional neural network,” IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524–2535, Dec. 2017.

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
        self.optimizer = setup_optimizer(args, self.model.parameters())
