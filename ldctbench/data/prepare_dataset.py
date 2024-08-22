import argparse
import os

import numpy as np
import pandas as pd
import pydicom
import yaml
from scipy.stats import iqr

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datafolder",
    default="/home/eliaseulig/Documents/Projects/explCT/LDCT_Mayo",
    help="Datafolder to be preprocessed by this script.",
)
parser.add_argument(
    "--train_frac", type=float, default=0.7, help="Fraction of training data."
)
parser.add_argument(
    "--val_frac", type=float, default=0.2, help="Fraction of validation data."
)
parser.add_argument(
    "--test_frac", type=float, default=0.1, help="Fraction of test data."
)
parser.add_argument(
    "--subset",
    type=float,
    default=1.0,
    help="Estimate on a subset of the trainingdata. Default = 1. (no subset)",
)
parser.add_argument(
    "--robust",
    action="store_true",
    help="If robust, use med, IQR and 95th percentile for max. Default = true",
)
parser.add_argument("--seed", type=int, default=1332, help="Seed to use")
opt = parser.parse_args()

np.random.seed(opt.seed)


def load_dicom(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices])
    return image


def generate_tvt_subsets(data, tvt_split=(0.7, 0.2, 0.1)):
    """
    Arguments:
        data:   Amount of samples (int) or list of samles (list)
        tvt_split:    Split ratio for training and validation

    Returns:
        Disjoint subsets of the set of all samples. If data is an int, we return indices, else sample sets
    """
    if not isinstance(data, int):
        num_data = len(data)
    else:
        num_data = data

    n_train = int(num_data * tvt_split[0])
    n_val = int(num_data * tvt_split[1])
    n_test = int(num_data * tvt_split[2])

    train_subset = list(np.random.choice(range(num_data), size=n_train, replace=False))
    val_subset = list(
        np.random.choice(
            list(set(range(num_data)) - set(train_subset)), size=n_val, replace=False
        )
    )
    test_subset = list(
        np.random.choice(
            list(set(range(num_data)) - set(train_subset) - set(val_subset)),
            size=n_test,
            replace=False,
        )
    )

    if isinstance(data, int):
        return train_subset, val_subset, test_subset

    return (
        [data[i] for i in train_subset],
        [data[i] for i in val_subset],
        [data[i] for i in test_subset],
    )


def get_statistics(samples, type):
    print("Load training stacks to estimate mean and std on {:s} ...".format(type))
    stacks_size = sum([512**2 * s["n_slices"] for s in samples])
    stacks = np.zeros(stacks_size, dtype="uint16")
    l_pnt = 0
    for i, s in enumerate(samples):
        print("Loaded stack {:d} of {:d} | {:s}".format(i + 1, len(samples), s[type]))
        r_pnt = l_pnt + 512**2 * s["n_slices"]
        stacks[l_pnt:r_pnt] = load_dicom(
            os.path.join(opt.datafolder, s[type][2:])
        ).flatten()
        l_pnt += 512**2 * s["n_slices"]

    min = float(np.min(stacks))
    if opt.robust:
        # We use 99.999 percentile to omit outliers
        max = float(np.quantile(stacks, 0.995))  # float(np.percentile(stacks, 0.99999))
        mean = float(np.median(stacks))
        std = float(iqr(stacks))
    else:
        max = float(np.max(stacks))
        mean = float(np.mean(stacks))
        std = float(np.std(stacks))

    return min, max, mean, std


def main():
    # Get training and validation set based on trainvarsplit
    splits = ["train", "val", "test"]
    subset = {split: [] for split in splits}
    metadata = pd.read_csv(os.path.join(opt.datafolder, "metadata.csv"))
    # Make Series description column lowercase and ignore GE data (no low dose available)
    metadata.drop(
        metadata[
            (metadata.Manufacturer == "GE")
            | (metadata.Manufacturer == "GE MEDICAL SYSTEMS")
        ].index,
        inplace=True,
    )
    metadata["Series Description"] = metadata["Series Description"].str.lower()

    full_dose = metadata.loc[metadata["Series Description"] == "full dose images"]
    low_dose = metadata.loc[metadata["Series Description"] == "low dose images"]
    patients = metadata["Subject ID"].unique()

    # Generate train, val, test split
    for exam_type in ["C", "L", "N"]:
        exam_subset = generate_tvt_subsets(
            data=[p for p in patients if p.startswith(exam_type)],
            tvt_split=(opt.train_frac, opt.val_frac, opt.test_frac),
        )
        for s, split in enumerate(splits):
            subset[split] += exam_subset[s]

    # Colllect metadata about each scan
    for split in splits:
        for i, patient in enumerate(subset[split]):
            n_images = full_dose.loc[full_dose["Subject ID"] == patient][
                "Number of Images"
            ].item()
            input_path = low_dose.loc[low_dose["Subject ID"] == patient][
                "File Location"
            ].item()
            target_path = full_dose.loc[full_dose["Subject ID"] == patient][
                "File Location"
            ].item()
            subset[split][i] = {
                "input": input_path,
                "target": target_path,
                "id": patient,
                "n_slices": n_images,
            }

    # Estimate only on a subset of training data if opt.subset <1:
    if opt.subset < 1:
        train_samples_est = np.random.choice(
            subset["train"], size=int(len(subset["train"]) * opt.subset), replace=False
        )
    else:
        train_samples_est = subset["train"]

    in_min, in_max, in_mean, in_std = get_statistics(train_samples_est, "input")

    print("Save info as yaml ...")
    with open(os.path.join(opt.datafolder, "info.yml"), "w") as outfile:
        yaml.dump(
            {
                "train_set": subset["train"],
                "val_set": subset["val"],
                "test_set": subset["test"],
                "mean": in_mean,
                "std": in_std,
                "min": in_min,
                "max": in_max,
                "robust": opt.robust,
            },
            outfile,
            default_flow_style=False,
        )


if __name__ == "__main__":
    main()
