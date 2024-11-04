import json
import os
import pickle
from argparse import Namespace
from typing import Tuple, Union

import numpy as np
import yaml


def load_yaml(path: str):
    """Load python object from yaml file"""
    with open(path) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content


def save_yaml(content, path: str):
    """Save python object to yaml file"""
    with open(path, "w") as file:
        yaml.dump(content, file)


def load_json(path: str):
    """Load python object from json file"""
    with open(path) as f:
        d = json.load(f)
        return d


def save_json(content, path: str):
    """Save python object to json file"""
    with open(path, "w") as f:
        json.dump(content, f)


def dump_config(args: Namespace, path: str):
    """Save argparse.Namespace to yaml file"""
    with open(os.path.join(path, "args.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def save_obj(obj, path: str):
    """Save python object to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path: str):
    """Load python object from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_center_width(
    x: np.ndarray,
    center: Union[int, float],
    width: Union[int, float],
    out_range: Tuple = (0, 1),
) -> np.ndarray:
    """Apply some center and width to an image

    Parameters
    ----------
    x : np.ndarray
        Image array of arbitrary dimension
    center : float
        Float indicating the center
    width : float
        Float indicating the width
    out_range : Tuple, optional
        Desired output range, by default (0, 1)

    Returns
    -------
    np.ndarray
        Copy of input array with center and width applied
    """
    center = float(center)
    width = float(width)
    lower = center - 0.5 - (width - 1) / 2.0
    upper = center - 0.5 + (width - 1) / 2.0
    res = np.empty(x.shape, dtype=x.dtype)
    res[x <= lower] = out_range[0]
    res[(x > lower) & (x <= upper)] = (
        (x[(x > lower) & (x <= upper)] - (center - 0.5)) / (width - 1.0) + 0.5
    ) * (out_range[1] - out_range[0]) + out_range[0]
    res[x > upper] = out_range[1]
    return res
