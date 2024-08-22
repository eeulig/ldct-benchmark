import copy
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from methods.base import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

from .network import Model, UNet
from .utils import SobelOperator, cutmix, ls_gan, mask_src_tgt, turn_on_spectral_norm


class Trainer(BaseTrainer):
    """Trainer for DUGAN[^1]

    [^1]: Z. Huang, J. Zhang, Y. Zhang, and H. Shan, “DU-GAN: Generative adversarial networks with dual-domain U-Net-based discriminators for low-dose CT denoising,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–12, 2022.
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

        # Setup models and their optimizers
        self.model = Model(args).to(self.dev)
        self.im_discriminator = UNet(
            repeat_num=6,
            use_discriminator=True,
            conv_dim=64,
            use_sigmoid=False,
        ).to(self.dev)
        self.im_discriminator = turn_on_spectral_norm(self.im_discriminator)
        self.grad_discriminator = copy.deepcopy(self.im_discriminator)

        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)
            self.im_discriminator = nn.DataParallel(
                self.im_discriminator, device_ids=self.args.devices
            )
            self.grad_discriminator = nn.DataParallel(
                self.grad_discriminator, device_ids=self.args.devices
            )

        self.g_optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.im_d_optimizer = optim.Adam(
            self.im_discriminator.parameters(), self.args.lr
        )
        self.grad_d_optimizer = optim.Adam(
            self.grad_discriminator.parameters(), self.args.lr
        )

        # Setup loss
        self.criterion = ls_gan

        # Setup sobel
        self.sobel = SobelOperator().to(self.dev)

        # Setup cutmix
        max_iter_upper = (
            self.args.max_iterations
            + self.args.iterations_before_val
            - (self.args.max_iterations % self.args.iterations_before_val)
        )
        self.apply_cutmix_prob = torch.rand(max_iter_upper)

        # Setup logging
        self.losses = utils.metrics.Losses(
            self.dataloader,
            [
                "D loss (img)",
                "D loss (grad)",
                "G loss (pix)",
                "G loss (grad)",
                "G loss",
            ],
        )

    def warmup(self):
        return min(
            self.iteration * self.args.cutmix_prob / self.args.cutmix_warmup_iter,
            self.args.cutmix_prob,
        )

    def train_discriminator(self, discriminator, optimizer, inputs, targets, fakes):
        optimizer.zero_grad()
        real_enc, real_dec = discriminator(targets)
        fake_enc, fake_dec = discriminator(fakes.detach())
        source_enc, source_dec = discriminator(inputs)

        d_loss = (
            self.criterion(real_enc, 1.0)
            + self.criterion(real_dec, 1.0)
            + self.criterion(fake_enc, 0.0)
            + self.criterion(fake_dec, 0.0)
            + self.criterion(source_enc, 0.0)
            + self.criterion(source_dec, 0.0)
        )

        apply_cutmix = self.apply_cutmix_prob[self.iteration - 1] < self.warmup()
        if apply_cutmix:
            mask = cutmix(real_dec.size()).to(real_dec)
            cutmix_enc, cutmix_dec = discriminator(
                mask_src_tgt(targets, fakes.detach(), mask)
            )
            cutmix_disc_loss = self.criterion(cutmix_enc, 0.0) + self.criterion(
                cutmix_dec, mask
            )
            cr_loss = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))
            d_loss += cutmix_disc_loss + cr_loss * self.args.lam_cutmix

        d_loss.backward()
        optimizer.step()

        return d_loss.data

    def train_step(self, batch):
        inputs, targets = batch["x"], batch["y"]
        self.iteration += 1

        gen_full_dose = self.model(inputs)
        grad_gen_full_dose = self.sobel(gen_full_dose)
        grad_low_dose = self.sobel(inputs)
        grad_full_dose = self.sobel(targets)

        # Train Discriminator
        for _ in range(self.args.n_d_train):
            im_d_loss = self.train_discriminator(
                self.im_discriminator,
                self.im_d_optimizer,
                inputs,
                targets,
                gen_full_dose,
            )

        # Train Generator
        self.g_optimizer.zero_grad()
        img_gen_enc, img_gen_dec = self.im_discriminator(gen_full_dose)
        img_gen_loss = self.criterion(img_gen_enc, 1.0) + self.criterion(
            img_gen_dec, 1.0
        )

        grad_d_loss = self.train_discriminator(
            self.grad_discriminator,
            self.grad_d_optimizer,
            grad_low_dose,
            grad_full_dose,
            grad_gen_full_dose,
        )
        grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_full_dose)
        grad_gen_loss = self.criterion(grad_gen_enc, 1.0) + self.criterion(
            grad_gen_dec, 1.0
        )

        # Pixelwise losses
        pix_loss = F.mse_loss(gen_full_dose, targets)
        grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

        # Sum up losses
        total_loss = (
            grad_gen_loss * self.args.lam_adv
            + img_gen_loss * self.args.lam_adv
            + pix_loss * self.args.lam_px_im
            + grad_loss * self.args.lam_px_grad
        )
        total_loss.backward()
        self.g_optimizer.step()

        self.losses.push(
            loss={
                "D loss (img)": im_d_loss,
                "D loss (grad)": grad_d_loss,
                "G loss (pix)": pix_loss.data,
                "G loss (grad)": grad_loss.data,
                "G loss": total_loss.data,
            },
            phase="train",
        )

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        inputs, targets = batch["x"], batch["y"]
        outputs = self.model(inputs)
        self.metrics.push(targets, outputs)

        im_d = self.im_discriminator(outputs)[1]
        grad_d = self.grad_discriminator(self.sobel(outputs))[1]

        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {
                    "low dose": inputs,
                    "high dose": targets,
                    "prediction": outputs,
                    "D_im(prediction)": im_d,
                    "D_grad(prediction)": grad_d,
                }
            )

    def train(self):
        self.model.train()
        self.im_discriminator.train()
        self.grad_discriminator.train()
        for batch in tqdm(self.dataloader["train"]):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.train_step(batch)

        self.losses.summarize("train")

    def validate(self):
        self.images = {}
        self.model.eval()
        self.im_discriminator.eval()
        self.grad_discriminator.eval()
        for batch_idx, batch in enumerate(tqdm(self.dataloader["val"])):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.val_step(batch_idx, batch)

        self.losses.summarize("val")
        self.metrics.summarize()
