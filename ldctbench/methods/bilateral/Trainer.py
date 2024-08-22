from argparse import Namespace

import torch
import torch.nn as nn
import utils
from methods.base import BaseTrainer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer class for trainable bilateral filter[^1]

    [^1]: F. Wagner et al., “Ultralow-parameter denoising: Trainable bilateral filter layers in computed tomography,” Medical Physics, vol. 49, no. 8, pp. 5107–5120, 2022.
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

        # As described in the paper, use different learning rates for sigma_xyz and sigma_r
        # sigma_xyz get the default learning rate args.lr, while sigma_r gets args.lr_r
        self.optimizer = utils.setup_optimizer(
            args,
            [
                {
                    "params": (
                        param
                        for name, param in self.model.named_parameters()
                        if "color" not in name
                    )
                },
                {
                    "params": (
                        param
                        for name, param in self.model.named_parameters()
                        if "color" in name
                    ),
                    "lr": args.lr_r,
                },
            ],
        )
