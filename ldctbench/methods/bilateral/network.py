import warnings
from argparse import Namespace

import torch
import torch.nn as nn
from bilateral_filter_layer import BilateralFilter3d


class Model(nn.Module):
    """Implements a model with trainable, bilateral filters"""

    def __init__(
        self, args: Namespace, sigma_xyz_init: float = 0.5, sigma_r_init: float = 0.01
    ):
        """Init function

        Parameters
        ----------
        args : Namespace
            Arguments passed to the model
        sigma_xyz_init : float, optional
            Initial value for sigma_xyz, by default 0.5
        sigma_r_init : float, optional
            Initial value for sigma_r, by default 0.01
        """
        super(Model, self).__init__()

        # If we have init values for sigma in args, then overwrite arguments given to module
        if hasattr(args, "sigma_xyz_init") and args.sigma_xyz_init is not None:
            warnings.warn(f"Using sigma_xyz_init={args.sigma_xyz_init} from args!")
        if hasattr(args, "sigma_r_init") and args.sigma_r_init is not None:
            warnings.warn(f"Using sigma_r_init={args.sigma_r_init} from args!")

        self.filters = nn.Sequential(
            BilateralFilter3d(
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_r_init,
                use_gpu=args.cuda,
            ),
            BilateralFilter3d(
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_r_init,
                use_gpu=args.cuda,
            ),
            BilateralFilter3d(
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_xyz_init,
                sigma_r_init,
                use_gpu=args.cuda,
            ),
        )

    def forward(self, x):
        # In our setup x has always shape (mbs, 1, N, H)
        x = torch.unsqueeze(x, dim=-1)
        x = self.filters(x)
        return x.squeeze(dim=-1)
