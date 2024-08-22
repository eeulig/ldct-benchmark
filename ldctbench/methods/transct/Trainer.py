from argparse import Namespace

import torch
import torch.nn as nn
import utils
from methods.base import BaseTrainer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer for TransCT[^1].

    [^1]: Z. Zhang, L. Yu, X. Liang, W. Zhao, and L. Xing, “TransCT: Dual-path transformer for low dose computed tomography,” in MICCAI, 2021.
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
