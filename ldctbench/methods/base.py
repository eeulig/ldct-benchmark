import os

import numpy as np
import torch
import wandb
from torch.autograd import Variable
from tqdm import tqdm

import ldctbench.utils as utils
from ldctbench.data.LDCTMayo import LDCTMayo


class BaseTrainer(object):
    def __init__(self, args, device):
        self.args = args
        self.dev = device

        # Setup datasets
        self.data = {phase: LDCTMayo(phase, self.args) for phase in ["train", "val"]}
        self.dataloader = utils.setup_dataloader(self.args, self.data)

        # Setup placeholders for loss, model, optimizer and losses
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.losses = utils.metrics.Losses(self.dataloader)
        self.metrics = utils.metrics.Metrics(
            self.dataloader,
            metrics=["SSIM", "PSNR", "RMSE"],
            denormalize_fn=self.data["train"].denormalize,
        )

        self.savedir = wandb.run.dir
        self.iteration = 0

    def train_step(self, batch):
        inputs, targets = batch["x"], batch["y"]

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1
        self.losses.push(loss, "train")

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        inputs, targets = batch["x"], batch["y"]

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.losses.push(loss, "val")
        self.metrics.push(targets, outputs)
        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {"low dose": inputs, "prediction": outputs, "high dose": targets}
            )

    def train(self):
        self.model.train()
        for batch in tqdm(self.dataloader["train"], desc="Train: "):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.train_step(batch)
        self.losses.summarize("train")

    def validate(self):
        self.images = {}
        self.model.eval()
        for batch_idx, batch in enumerate(
            tqdm(self.dataloader["val"], desc="Validate: ")
        ):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.val_step(batch_idx, batch)
        self.losses.summarize("val")
        self.metrics.summarize()

    def save_checkpoint(self, to_optimize="SSIM", minimize=False):
        if to_optimize in self.metrics.names:
            values = self.metrics.metrics[to_optimize]
        elif to_optimize in self.losses.names:
            values = self.losses.losses["val"][to_optimize]
        else:
            raise ValueError(
                f"to_optimize must be logged by metrics or losses but got {to_optimize} instead"
            )

        find_opt = np.argmin if minimize else np.argmax

        if find_opt(values) == len(values) - 1:
            # Last value was the best one so far
            print(
                f"Store network at iteration {self.iteration} with {to_optimize}: {values[-1]}"
            )
            checkpoint_path = os.path.join(self.savedir, f"best_{to_optimize}.pt")
            state_dict = (
                self.model.module.state_dict()
                if isinstance(self.args.devices, list)
                else self.model.state_dict()
            )
            torch.save(
                {
                    "args": self.args,
                    "iteration": self.iteration,
                    "model_state_dict": state_dict,
                },
                checkpoint_path,
            )

    def log(self):
        self.save_checkpoint()
        # Losses
        self.losses.log(self.savedir, self.iteration, self.args.iterations_before_val)
        self.losses.plot(self.savedir)
        # Metrics
        self.metrics.log(self.savedir, self.iteration, self.args.iterations_before_val)
        self.metrics.plot(self.savedir)
        # Validation samples
        wandb.log(self.images, step=self.iteration)

    def log_wandb_images(self, images):
        for tag, img in images.items():
            img = wandb.Image(img.data.cpu()[0], caption=tag)
            if tag not in self.images:
                self.images[tag] = [img]
            else:
                self.images[tag].append(img)

    def fit(self):
        delta_seed = 0
        while self.iteration < self.args.max_iterations:
            torch.manual_seed(self.args.seed + delta_seed)
            np.random.seed(self.args.seed + delta_seed)

            # Train and validate
            self.train()
            self.validate()

            # Log
            self.log()
            self.losses.reset()
            self.metrics.reset()

            delta_seed += 1
