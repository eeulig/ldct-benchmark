import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--cutmix_warmup_iter", type=int, help="Number of warmup iterations for cutmix"
    )
    parser.add_argument("--cutmix_prob", type=float, help="Cutmix probability")
    parser.add_argument("--lam_adv", type=float, help="Adv. weight in generator loss")
    parser.add_argument(
        "--lam_px_im", type=float, help="Pixelwise loss of image in generator loss"
    )
    parser.add_argument(
        "--lam_px_grad", type=float, help="Pixelwise loss of gradient in generator loss"
    )
    parser.add_argument(
        "--lam_cutmix", type=float, help="Weight of cutmix loss in distriminator loss"
    )
    return parser
