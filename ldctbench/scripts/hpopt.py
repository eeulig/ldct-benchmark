import argparse

import wandb
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="Config file to use for hyperparameter optimization",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        help="Number of steps of hyperparameter optimization",
    )
    opt = parser.parse_args()

    # Run hyperparameter optimization
    with open(opt.config, "r") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)  # Load sweep config
    sweep_id = wandb.sweep(sweep_config)  # Initialize the sweep
    print(
        f"Starting hyperparameter optimization with {opt.n_runs} runs. Sweep ID: {sweep_id}"
    )
    wandb.agent(sweep_id, count=opt.n_runs)  # Start the agent

    # After sweep is finished, find network with best performance
    api = wandb.Api()
    sweep = api.sweep(f"ldct-benchmark/{sweep_id}")
    best_run = sweep.best_run()
    print(
        f"\033[1mSummary of hyperparameter optimization with sweep ID \033[92m{sweep_id}\033[0m"
    )
    print(f"   Number of runs: {len(sweep.runs)}")
    print(f"   Best run: \033[92m{best_run.name}\033[0m")
    print(f"   SSIM: {best_run.summary['SSIM']:.3f}")
    print(f"   PSNR: {best_run.summary['PSNR']:.3f}")
    print(f"   RMSE: {best_run.summary['RMSE']:.3f}")


if __name__ == "__main__":
    main()
