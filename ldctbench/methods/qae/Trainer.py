from argparse import Namespace

import torch
import torch.nn as nn
import utils
from methods.base import BaseTrainer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer class for Quadratic Autoencoder (QAE)[^1].

    [^1]: F. Fan et al., “Quadratic autoencoder (Q-AE) for low-dose CT denoising,” IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 2035–2050, Jun. 2020.
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
