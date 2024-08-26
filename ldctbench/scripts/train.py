import glob
import importlib
import os
import shutil
import time
import warnings

import matplotlib
import numpy as np
import torch
import wandb

import ldctbench.utils.auxiliaries as aux
from ldctbench.utils.argparser import make_parser, use_config

matplotlib.use("Agg")

os.environ["WANDB_START_METHOD"] = (
    "thread"  # Necessary to spawn subprocess on cluster node
)
os.environ["WANDB_IGNORE_GLOBS"] = (
    "*.pt"  # Do not upload network models to wandb to save space
)


def train(args):
    # Setup seeds
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device(s) to use
    if isinstance(args.devices, list) and len(args.devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.devices])
        args.devices = list(range(len(args.devices)))
    elif isinstance(args.devices, list) and len(args.devices) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
        args.devices = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        args.devices = 0
    device = torch.device("cuda" if args.cuda else "cpu")

    # Setup wandb. Without this hack training may abort if network connection is interrupted
    while True:
        try:
            if hasattr(args, "wandbtag") and args.wandbtag:
                wandb.init(project="ldct-benchmark", config=args, tags=[args.wandbtag])
            else:
                wandb.init(project="ldct-benchmark", config=args)
            break
        except Exception as e:
            print(f"{e} ... retrying...")
            time.sleep(10)

    wandb.run.name = wandb.run.dir.split(os.sep)[-2].split("run-")[-1]
    aux.dump_config(args, wandb.run.dir)

    # Setup trainer
    try:
        trainer_module = importlib.import_module(
            "ldctbench.methods.{}.Trainer".format(args.trainer)
        )
    except ModuleNotFoundError:
        raise ValueError(
            "Trainer {0} not known and module methods.{0}.Trainer not found".format(
                args.trainer
            )
        )
    trainer_class = getattr(trainer_module, "Trainer")
    trainer = trainer_class(args, device)

    # Train model
    print("Start training...")
    trainer.fit()


def main():
    parser = make_parser()
    args = parser.parse_args()
    args = use_config(args)
    if not hasattr(args, "datafolder") or not args.datafolder:
        if "LDCTBENCH_DATAFOLDER" in os.environ:
            args.datafolder = os.environ["LDCTBENCH_DATAFOLDER"]
            warnings.warn(
                f"No datafolder in args. Will use the one provided via environment variable LDCTBENCH_DATAFOLDER: {args.datafolder}"
            )
        else:
            raise ValueError(
                "No datafolder provided! Add via\n \t- Config file: add key: datafolder\n\t- Arguments: add argument --datafolder\n\t- Environment variable: export LDCTBENCH_DATAFOLDER=..."
            )

    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"

    train(args)


if __name__ == "__main__":
    main()
