import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--n_d_train", type=int, help="Number of times to train D for each G iter"
    )
    parser.add_argument(
        "--lam_perc", type=float, help="Weight of VGG (perceptual) loss"
    )
    return parser
