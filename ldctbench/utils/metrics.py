import csv
import os
from typing import Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from skimage.metrics import mean_squared_error, structural_similarity

MAX_DATA_VAL = 2924.0
EPS = 1e-8


class Losses(object):
    """Object to log losses

    Examples
    --------
    In a training, losses can be logged as follows:

    >>> from ldctbench.utils.metrics import Losses
    >>> # Setup training
    >>> dataloader = {"train": DataLoader(Dataset("train"), ...), "val": DataLoader(Dataset("val"), ...)}
    >>> losses = Losses(dataloader)
    >>> # Perform training and validation routine
    >>> for epoch in n_epochs:
    >>>     # Train model
    >>>     for batch in dataloader["train"]:
    >>>         x, y = batch
    >>>         y_hat = model(x)
    >>>         loss = criterion(y_hat, y)
    >>>         # Perform weight optim
    >>          ...
    >>>         losses.push(loss, "train")
    >>>     losses.summarize("train")
    >>>     # Validate model
    >>>     for batch in dataloader["val"]:
    >>>         x, y = batch
    >>>         y_hat = model(x)
    >>>         loss = criterion(y_hat, y)
    >>>         losses.push(loss, "val")
    >>>     losses.summarize("val")
    >>> # Log
    >>> losses.log(savedir, epoch, 0)
    >>> losses.plot(savedir)
    """

    def __init__(
        self, dataloader: Dict[str, torch.utils.data.DataLoader], losses: str = "loss"
    ):
        """Init function

        Parameters
        ----------
        dataloader : Dict[str, torch.utils.data.DataLoader]
            Dict containing the dataloaders
        losses : str, optional
            Name of losses, by default "loss"
        """
        self.dataloader = dataloader
        self.names = losses if isinstance(losses, list) else [losses]
        self.losses = {
            phase: {loss: [0.0] for loss in self.names} for phase in dataloader.keys()
        }

    def push(
        self,
        loss: Union[Dict[str, torch.Tensor], torch.Tensor],
        phase: str,
        name: Optional[str] = "loss",
    ):
        """Push loss to object

        Parameters
        ----------
        loss : Union[Dict[str, torch.Tensor], torch.Tensor]
            Single loss or dict of losses
        phase : str
            To which phase the loss(es) to push belongs
        name : str, optional
            Name of loss (only necessary if loss is not a dict), by default "loss"
        """
        if isinstance(loss, dict):
            for n, l in loss.items():
                self.losses[phase][n][-1] += l.item()
        else:
            self.losses[phase][name][-1] += loss.item()

    def summarize(self, phase: str):
        """Summarize losses for this epoch

        Parameters
        ----------
        phase : str
            For which phase to summarize loss
        """
        for name in self.names:
            self.losses[phase][name][-1] /= len(self.dataloader[phase])

    def reset(self):
        """Reset losses (add new epoch)"""
        for phase in self.losses:
            for name in self.names:
                self.losses[phase][name].append(0.0)

    def log(self, savepath: str, iteration: int, iterations_before_val: int):
        """Log losses to wandb and local file

        Parameters
        ----------
        savepath : str
            Where to store losses .csv file
        iteration : int
            Current iteration
        iterations_before_val : int
            Number of of training iterations before validation
        """

        # Log to wandb
        wandb.log(
            {
                "{} ({})".format(name, phase): loss[-1]
                for phase, losses in self.losses.items()
                for name, loss in losses.items()
            },
            step=iteration,
        )

        # Log to file
        with open(os.path.join(savepath, "Losses.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if iteration == iterations_before_val:
                writer.writerow(
                    ["Iteration"]
                    + [
                        "{} {}".format(name, phase)
                        for name in self.names
                        for phase in self.losses
                    ]
                )
            writer.writerow(
                [iteration]
                + [
                    loss[name][-1]
                    for name in self.names
                    for loss in self.losses.values()
                ]
            )

    def plot(self, savepath: str, y_log: bool = True):
        """Plot losses to a file

        Parameters
        ----------
        savepath : str
            WHere to store pdf file
        y_log : bool, optional
            Plot y axis in logarithmic scale, by default True
        """
        for name in self.names:
            plt.figure(figsize=(5, 3))
            for phase in self.losses.keys():
                y = self.losses[phase][name]
                x = [(i + 1) * len(self.dataloader["train"]) for i in range(len(y))]
                plt.plot(x, y, label=phase)
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                if y_log:
                    plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Losses_{}.pdf".format(name)))
            plt.close("all")


class Metrics(object):
    """Object to log metrics

    Examples
    --------
    In a training, metrics can be logged as follows:

    >>> from ldctbench.utils.metrics import Metrics
    >>> # Setup training
    >>> dataloader = {"train": DataLoader(Dataset("train"), ...), "val": DataLoader(Dataset("val"), ...)}
    >>> metrics = Metrics(dataloader, metrics=["SSIM", "PSNR", "RMSE"])
    >>> # Perform training and validation routine
    >>> for epoch in n_epochs:
    >>>     # Train model
    >>>     ...
    >>>     # Validate model
    >>>     for batch in dataloader["val"]:
    >>>         x, y = batch
    >>>         y_hat = model(x)
    >>>         metrics.push(y_hat, y)
    >>>     metrics.summarize()
    >>> # Log
    >>> metrics.log(savedir, epoch,0)
    >>> metrics.plot(savedir)
    """

    def __init__(
        self,
        dataloader: Dict[str, torch.utils.data.DataLoader],
        metrics: str,
        denormalize_fn: Optional[Callable] = None,
    ):
        """Init function

        Parameters
        ----------
        dataloader : Dict[str, torch.utils.data.DataLoader]
            Dict containing the dataloaders
        metrics : str
            Name of metrics to log. Must be RMSE | SSIM | PSNR
        denormalize_fn : Optional[Callable], optional
            Function to use for denormalizing images before computing metrics, by default None
        """
        self.dataloader = dataloader
        self.names = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = {metric: [0.0] for metric in self.names}
        self.denormalize_fn = denormalize_fn

    def push(
        self,
        targets: Union[np.ndarray, torch.Tensor],
        predictions: Union[np.ndarray, torch.Tensor],
    ):
        """Compute metrics for given targets and predictions

        Parameters
        ----------
        targets : Union[np.ndarray, torch.Tensor]
            Ground truth reference
        predictions : Union[np.ndarray, torch.Tensor]
            Prediction by the network

        Raises
        ------
        ValueError
            If shape of predictions and targets are not identical
        ValueError
            If metric provided in init function is not in SSIM | PSNR | RMSE
        """
        if isinstance(targets, torch.Tensor):
            if targets.is_cuda:
                targets = targets.data.cpu().numpy()
            else:
                targets = targets.data.numpy()
        if isinstance(predictions, torch.Tensor):
            if predictions.is_cuda:
                predictions = predictions.data.cpu().numpy()
            else:
                predictions = predictions.data.numpy()

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape of predictions and targets must be identical but got {predictions.shape} and {targets.shape}!"
            )

        # Denormalize and clip to maximum HU of bone (1900) + offset (1024) -> 2924
        if self.denormalize_fn:
            targets = self.denormalize_fn(targets)
            predictions = self.denormalize_fn(predictions)
            targets = np.clip(targets, a_min=0.0, a_max=MAX_DATA_VAL)
            predictions = np.clip(predictions, a_min=0.0, a_max=MAX_DATA_VAL)

        # Iterate over images in batch
        n_imgs = predictions.shape[0]
        for metric in self.names:
            avg_metric = 0.0
            for im_idx in range(n_imgs):
                t = targets[im_idx].squeeze()
                p = predictions[im_idx].squeeze()

                if metric == "SSIM":
                    avg_metric += structural_similarity(t, p, data_range=MAX_DATA_VAL)
                elif metric == "PSNR":
                    err = mean_squared_error(t, p)
                    avg_metric += 10 * np.log10((MAX_DATA_VAL**2) / (err + EPS))
                elif metric == "RMSE":
                    avg_metric += np.sqrt(mean_squared_error(t, p))
                else:
                    raise ValueError(f"Unknown metric {metric}")
            self.metrics[metric][-1] += avg_metric / n_imgs

    def summarize(self):
        """Summarize metric for this epoch"""
        for name in self.names:
            self.metrics[name][-1] /= len(self.dataloader["val"])

    def reset(self):
        """Reset metrics (start new epoch)"""
        for name in self.names:
            self.metrics[name].append(0.0)

    def log(self, savepath: str, iteration: int, iterations_before_val: int):
        """Log metrics to wandb and local file

        Parameters
        ----------
        savepath : str
            Where to store losses .csv file
        iteration : int
            Current iteration
        iterations_before_val : int
            Number of of training iterations before validation
        """
        # Log to wandb
        wandb.log(
            {"{}".format(name): metric[-1] for name, metric in self.metrics.items()},
            step=iteration,
        )

        # Log to file
        with open(os.path.join(savepath, "Metrics.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if iteration == iterations_before_val:
                writer.writerow(
                    ["Iteration"] + ["{}".format(name) for name in self.names]
                )
            writer.writerow(
                [iteration] + [self.metrics[name][-1] for name in self.names]
            )

    def plot(self, savepath: str):
        """Plot metrics to a file

        Parameters
        ----------
        savepath : str
            WHere to store pdf file
        """

        for name in self.names:
            plt.figure(figsize=(5, 3))
            y = self.metrics[name]
            x = [(i + 1) * len(self.dataloader["train"]) for i in range(len(y))]
            plt.plot(x, y)
            plt.xlabel("Iteration")
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Metrics_{}.pdf".format(name)))
            plt.close("all")
