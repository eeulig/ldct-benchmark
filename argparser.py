import argparse
import importlib

import yaml

METHODS = [
    "bilateral",
    "cnn10",
    "dugan",
    "qae",
    "redcnn",
    "resnet",
    "transct",
    "wganvgg",
]


def make_parser():
    parser = argparse.ArgumentParser()

    # -----------------------------   General Setup   ------------------------------#
    parser.add_argument("--config", default="", help="Use yaml file as config.")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Use cuda")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="Use cuda")
    parser.set_defaults(cuda=True)
    parser.add_argument(
        "--devices", nargs="+", type=int, default=[0, 1], help="Cuda device(s) to use"
    )
    parser.add_argument("--trainer", help="Use this trainer")
    parser.add_argument("--mbs", type=int, help="Minibatch size")
    parser.add_argument("--num_workers", type=int, help="Number of workers to use")
    parser.add_argument("--valsamples", type=int, help="How many val samples to log")
    parser.add_argument(
        "--dryrun", action="store_true", help="Deactivate syncing to Weights & Biases"
    )
    parser.add_argument("--wandbtag", default="", help="W&B Tag (default = no tag)")
    # --------------------------------   Training   --------------------------------#
    parser.add_argument(
        "--max_iterations", type=int, help="Maximum number of iterations to train"
    )
    parser.add_argument(
        "--iterations_before_val",
        type=int,
        help="Number of iterations before we validate",
    )
    parser.add_argument("--optimizer", help="Optimizer to use.")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--adam_b1",
        type=float,
        default=0.9,
        help="b1 parameter of Adam (default = 0.9)",
    )
    parser.add_argument(
        "--adam_b2",
        type=float,
        default=0.999,
        help="b2 parameter of Adam (default = 0.999)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use for the optimizer.",
    )

    # ----------------------------------   Data   ----------------------------------#
    parser.add_argument(
        "--datafolder",
        default="",
        help="Path to datafolder",
    )
    parser.add_argument(
        "--data_norm",
        default="meanstd",
        help="Input normalization: minmax | meanstd (default = meanstd)",
    )
    parser.add_argument(
        "--data_subset",
        type=float,
        default=1.0,
        help="Layer to extract activations from (default = 15)",
    )
    parser.add_argument(
        "--patchsize",
        type=int,
        help="Patchsize used for training If 0, the full image is used instead.",
    )
    parser.add_argument(
        "--eval_patchsize",
        type=int,
        default=128,
        help="Patchsize used for Evaluation If 0, the full image is used instead.",
    )
    # ------------------------------   Random Seeds   ------------------------------#
    parser.add_argument("--seed", default=1332, type=int, help="manual seed")

    # Add arguments from individual methods
    for method_name in METHODS:
        argparse_mod = importlib.import_module(
            f"ldctbench.methods.{method_name}.argparser"
        )
        add_args = getattr(argparse_mod, "add_args")
        parser = add_args(parser)

    return parser


def use_config(args):
    if args.config:
        file = open(args.config)
        parsed_yaml = yaml.load(file, Loader=yaml.FullLoader)
        args = argparse.Namespace(**parsed_yaml)
    return args
