from argparse import Namespace
from typing import Dict, Iterator, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def setup_dataloader(
    args: Namespace, datasets: Dict[str, Dataset]
) -> Dict[str, DataLoader]:
    """Returns dict of dataloaders

    Parameters
    ----------
    args : Namespace
        Command line arguments
    datasets : Dict[str, Dataset]
        Dictionary of datasets for each phase.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionray of dataloaders for each phase.
    """
    dataloader = {}
    for phase, data in datasets.items():
        num_samples = (
            args.mbs * args.iterations_before_val if phase == "train" else len(data)
        )
        sampler = WeightedRandomSampler(data.weights, num_samples, replacement=True)
        dataloader[phase] = DataLoader(
            dataset=data,
            batch_size=args.mbs,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
            drop_last=True,
            sampler=sampler,
        )
    return dataloader


def setup_optimizer(
    args: Namespace, parameters: Iterator[nn.parameter.Parameter]
) -> optim.Optimizer:
    """Setup optimizer for given model parameters

    Parameters
    ----------
    args : Namespace
        Command line arguments
    parameters : Iterator[nn.parameter.Parameter]
        Parameters to be optimized. For some `model: nn.Module` these can be received via `model.parameters()`

    Returns
    -------
    optim.Optimizer
        Optimizer for the given parameters

    Raises
    ------
    ValueError
        If args.optimizer is not in "sgd" | "adam" | "rmsprop"
    """
    if args.optimizer == "sgd":
        return optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.sgd_momentum,
            dampening=args.sgd_dampening,
        )
    elif args.optimizer == "adam":
        return optim.Adam(parameters, lr=args.lr, betas=(args.adam_b1, args.adam_b2))
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(parameters, lr=args.lr)
    else:
        raise ValueError("Optimizer unknown. Must be one of sgd | adam | rmsprop")


class repeat_ch(object):
    """Class to repeat input 3 times in channel dimension if `in_ch == 1`"""

    def __init__(self, in_ch: int):
        """Init funciton

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        """
        self.in_ch = in_ch

    def __call__(self, x):
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        network: str,
        device: torch.device,
        in_ch: int = 3,
        layers: List[int] = [3, 8, 15, 22],
        norm: str = "l1",
        return_features: bool = False,
    ):
        """Perceptual Loss used for the WGAN-VGG training

        The `layers` argument defines where to extract the activations. In the default paper, style losses are computed
        at: `3: "relu1_2"`, `8: "relu2_2"`, `15: "relu3_3"`, `22: "relu4_3"` and perceptual (content) loss is evaluated at: `15: "relu3_3"`. In[^1] the content is evaluated in vgg19 after the 16th (last) conv layer (layer `34`)

        [^1]: Q. Yang et al., “Low-dose CT image denoising using a generative adversarial network with wasserstein distance and perceptual loss,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1348–1357, Jun. 2018.

        Parameters
        ----------
        network : str
            Which VGG flavor to use, must be "vgg16" or "vgg19"
        device : torch.device
            Torch device to use
        in_ch : int, optional
            Number of input channels, by default 3
        layers : List[int], optional
            Number of layers at which to extract features, by default [3, 8, 15, 22]
        norm : str, optional
            Pixelwise, must be "l1" or "mse", by default "l1"
        return_features : bool, optional
            _description_, by default False

        Raises
        ------
        ValueError
            `norm` is neither "l1" nor "mse".
        """
        super(PerceptualLoss, self).__init__()

        if network == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        elif network == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        vgg.eval()

        self.vgg_features = vgg.features
        self.layers = [str(layer) for layer in layers]
        if norm == "l1":
            self.norm = nn.L1Loss()
        elif norm == "mse":
            self.norm = nn.MSELoss()
        else:
            raise ValueError("Norm {} not known for PerceptualLoss".format(norm))
        self.transform = repeat_ch(in_ch)
        self.return_features = return_features

    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)

        loss = 0.0
        if self.return_features:
            features = {"input": [], "target": []}

        for i, m in self.vgg_features._modules.items():
            input = m(input)
            target = m(target)

            if i in self.layers:
                loss += self.norm(input, target)
                if self.return_features:
                    features["input"].append(input.clone())
                    features["target"].append(target.clone())

                if i == self.layers[-1]:
                    break

        return (loss, features) if self.return_features else loss
