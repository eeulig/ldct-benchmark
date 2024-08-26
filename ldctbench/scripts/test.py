import argparse
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

from ldctbench.data import TestData
from ldctbench.hub import Methods, load_model
from ldctbench.utils import compute_metric, save_raw, save_yaml, setup_trained_model


def main():
    # Commandline Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "cnn10",
            "redcnn",
            "wganvgg",
            "resnet",
            "qae",
            "dugan",
            "transct",
            "bilateral",
        ],
        help="Methods to evaluate. Can be either method name of pretrained networks (cnn10, redcnn, qae, ...), or run name of network trained by the user.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["SSIM", "PSNR", "VIF"],
        help="Metrics to compute. Must be either SSIM, PSNR, RMSE, or VIF.",
    )
    parser.add_argument(
        "--datafolder",
        default="",
        help="Path to datafolder. If not set, an environemnt variable LDCTBENCH_DATAFOLDER must be set.",
    )
    parser.add_argument(
        "--data_norm",
        default="meanstd",
        help="Input normalization: Must be either minmax or meanstd.",
    )
    parser.add_argument(
        "--results_dir",
        default="./results/",
        help="Folder where to store results.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id.",
    )
    parser.add_argument(
        "--print_table",
        dest="print_table",
        action="store_true",
        help="Print results in table.",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    DEV = torch.device("cuda:0")

    # Setup data
    if not args.datafolder:
        if "LDCTBENCH_DATAFOLDER" in os.environ:
            args.datafolder = os.environ["LDCTBENCH_DATAFOLDER"]
            warnings.warn(
                f"No datafolder in args. Will use the one provided via environment variable LDCTBENCH_DATAFOLDER: {args.datafolder}"
            )
        else:
            raise ValueError(
                "No datafolder provided! Add via\n \t- Arguments: add argument --datafolder\n\t- Environment variable: export LDCTBENCH_DATAFOLDER=..."
            )

    data = TestData(args.datafolder, args.data_norm)

    # Collect networks
    networks = {}
    for method in args.methods:
        if method in [m.value for m in Methods]:
            networks[method] = load_model(method, eval=True).to(DEV)
        else:
            net = setup_trained_model(
                run_name=method,
                device=DEV,
                network_name="Model",
                state_dict="best_SSIM",
                eval=True,
            )
            networks[method] = net

    # Setup metrics dict: patient -> metric -> method
    metrics = {
        pat["info"]["id"]: {
            m: {method: [] for method in ["LD"] + list(networks)} for m in args.metrics
        }
        for pat in data.samples
    }

    with torch.no_grad():
        for patient in (pbar := tqdm(data, desc="Inference for patient: ")):
            patient_name = patient["info"]["id"]
            f_hd = patient["f_hd"]
            f_ld = patient["f_ld"]
            exam_type = f_hd[0][0].split("/")[1][0]

            # Save HD and LD
            savedir = os.path.join(args.results_dir, patient_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            save_raw(
                savedir,
                "LD",
                data._convert_hu(
                    data.denormalize(patient["x"]).squeeze().numpy(), to_hu=True
                ),
            )
            save_raw(
                savedir,
                "HD",
                data._convert_hu(
                    data.denormalize(patient["y"]).squeeze().numpy(), to_hu=True
                ),
            )

            # Apply networks
            for i_method, (method_name, method) in enumerate(networks.items()):
                pbar.set_description(f"Inference for {patient_name} ({method_name})")
                predictions = []
                for slice in range(patient["info"]["n_slices"]):
                    x = torch.unsqueeze(torch.unsqueeze(patient["x"][slice], 0), 0).to(
                        DEV
                    )
                    y = torch.unsqueeze(torch.unsqueeze(patient["y"][slice], 0), 0).to(
                        DEV
                    )

                    # Compute metrics for LD image
                    if i_method == 0:
                        res = compute_metric(
                            y,
                            x,
                            metrics=args.metrics,
                            denormalize_fn=data.denormalize,
                            exam_type=exam_type,
                        )
                        for m, r in res.items():
                            metrics[patient_name][m]["LD"].extend(r)

                    y_hat = method(x)
                    res = compute_metric(
                        y,
                        y_hat,
                        metrics=args.metrics,
                        denormalize_fn=data.denormalize,
                        exam_type=exam_type,
                    )
                    for m, r in res.items():
                        metrics[patient_name][m][method_name].extend(r)
                    predictions.append(
                        data.denormalize(y_hat.squeeze().detach().cpu().numpy())
                    )

                # Save prediction to raw
                save_raw(
                    savedir,
                    method_name,
                    data._convert_hu(
                        np.stack(predictions, axis=0),
                        to_hu=True,
                    ),
                )

    # Save test results
    save_yaml(metrics, os.path.join(args.results_dir, "test_metrics.yaml"))

    if args.print_table:
        # Aggregate in dataframe
        results_df = pd.DataFrame.from_records(
            [
                (patient[0], patient, metric, method, image)
                for patient, metric_dict in metrics.items()
                for metric, method_dict in metric_dict.items()
                for method, images in method_dict.items()
                for image in images
            ],
            columns=["exam_type", "patient", "metric", "method", "value"],
        )

        # Print results as markdown table
        exam_types = {"C": "Chest", "L": "Abdomen", "N": "Neuro"}
        results_avg = {
            "Method": ["LD"] + args.methods,
            **{
                f"{metric} ({exam_types[et]})": [
                    np.round(
                        results_df[
                            (results_df.metric == metric)
                            & (results_df.method == method)
                            & (results_df.exam_type == et)
                        ].value.mean(),
                        3,
                    )
                    for method in ["LD"] + args.methods
                ]
                for metric, et in product(args.metrics, exam_types.keys())
            },
        }
        print(tabulate(results_avg, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
