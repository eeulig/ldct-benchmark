from argparse import Namespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import utils
from methods.base import BaseTrainer

from .network import Discriminator, Model


class Trainer(BaseTrainer):
    """Trainer for WGAN-VGG[^1]

    [^1]: Q. Yang et al., “Low-dose CT image denoising using a generative adversarial network with wasserstein distance and perceptual loss,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1348–1357, Jun. 2018.
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

        # Setup model and its optimizer
        self.model = Model(self.args).to(self.dev)
        self.critic = Discriminator(input_size=args.patchsize).to(self.dev)

        # Setup perceptual loss
        self.perceptual = utils.PerceptualLoss(
            network="vgg19", device=self.dev, layers=["34"], in_ch=1, norm="l1"
        )

        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)
            self.critic = nn.DataParallel(self.critic, device_ids=self.args.devices)

        self.g_optimizer = utils.setup_optimizer(args, self.model.parameters())
        self.d_optimizer = utils.setup_optimizer(args, self.critic.parameters())

        # Setup logging
        self.losses = utils.metrics.Losses(
            self.dataloader, ["D loss", "G loss adv", "grad. pen.", "G loss perc."]
        )

    def gradient_penalty(
        self, target: torch.Tensor, fake: torch.Tensor, lam: float = 10.0
    ) -> torch.Tensor:
        """Compute gradient penalty for given target and fake.

        Parameters
        ----------
        target : torch.Tensor
            Ground truth tensor
        fake : torch.Tensor
             Fake = G(x) tensor
        lam : float, optional
            lambda to weigh gradient penalty, by default 10.

        Returns
        -------
        torch.Tensor
            Computed penalty using provided target, fake and self.critic
        """
        assert target.size() == fake.size()
        a = torch.FloatTensor(np.random.random((target.size(0), 1, 1, 1))).to(self.dev)
        interp = (a * target + ((1 - a) * fake)).requires_grad_(True)
        d_interp = self.critic(interp)
        fake_ = (
            torch.FloatTensor(target.shape[0], 1)
            .fill_(1.0)
            .to(self.dev)
            .requires_grad_(False)
        )
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=fake_,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam
        return gradient_penalty

    def train_step(self, batch: Dict[str, torch.Tensor]):
        """Training step

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch coming from training dataloader containing LD input and HD ground truth.
        """
        inputs, targets = batch["x"], batch["y"]

        #  Train discriminator
        self.d_optimizer.zero_grad()
        for _ in range(self.args.n_d_train):
            fakes = self.model(inputs)
            critic_loss = torch.mean(self.critic(fakes)) - torch.mean(
                self.critic(targets)
            )
            grad_p = self.gradient_penalty(targets, fakes)
            loss_D = critic_loss + grad_p

            loss_D.backward()
            self.d_optimizer.step()

        #  Train Generator
        self.g_optimizer.zero_grad()

        fakes = self.model(inputs)
        loss_G_adv = -torch.mean(self.critic(fakes))
        loss_G_perc = self.perceptual(fakes, targets)
        loss_G = (self.args.lam_perc * loss_G_perc) + loss_G_adv

        loss_G.backward()
        self.g_optimizer.step()

        self.iteration += 1
        self.losses.push(
            loss={
                "D loss": loss_D.data,
                "G loss adv": loss_G.data,
                "G loss perc.": loss_G_perc.data,
                "grad. pen.": grad_p.data,
            },
            phase="train",
        )

    @torch.no_grad()
    def val_step(self, batch_idx: int, batch: Dict[str, torch.Tensor]):
        """Validation step

        Parameters
        ----------
        batch_idx: int
            Batch idx necessary for logging of samples.
        batch : Dict[str, torch.Tensor]
            Batch coming from validation dataloader containing LD input and HD ground truth.
        """

        inputs, targets = batch["x"], batch["y"]
        outputs = self.model(inputs)

        self.metrics.push(targets, outputs)

        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {"low dose": inputs, "prediction": outputs, "high dose": targets}
            )
