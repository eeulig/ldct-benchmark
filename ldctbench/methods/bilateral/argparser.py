import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--lr_r", type=float, help="Learning rate for sigma_r")
    parser.add_argument(
        "--sigma_xyz_init", type=float, help="Initialization for sigma_xyz"
    )
    parser.add_argument("--sigma_r_init", type=float, help="Initialization for sigma_r")
    return parser
