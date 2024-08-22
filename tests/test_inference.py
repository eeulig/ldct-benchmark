import os

import numpy as np

import ldctbench
from ldctbench.hub import Methods
from ldctbench.utils import CW, apply_center_width, load_yaml, normalize

package_dir = os.path.join(os.path.dirname(os.path.abspath(ldctbench.__file__)))

DATA_INFO = load_yaml(os.path.join(package_dir, "data", "info.yml"))


def test_normalize_method_then_normalize_meanstd():
    # Since all methods are trained with meanstd normalization
    x = np.random.randn(10, 10)
    mean_ = DATA_INFO["mean"]
    std_ = DATA_INFO["std"]
    x_t = normalize(x, method=Methods.CNN10)
    assert np.all(np.isclose((x - mean_) / std_, x_t))


def test_normalize_meanstd_then_normalize_correct():
    x = np.random.randn(10, 10)
    mean_ = DATA_INFO["mean"]
    std_ = DATA_INFO["std"]
    x_t = normalize(x, normalization="meanstd")
    assert np.all(np.isclose((x - mean_) / std_, x_t))


def test_normalize_minmax_then_normalize_correct():
    x = np.random.randn(10, 10)
    min_ = DATA_INFO["min"]
    max_ = DATA_INFO["max"]
    x_t = normalize(x, normalization="minmax")
    assert np.all(np.isclose((x - min_) / (max_ - min_), x_t))


def test_apply_center_width_in_range():
    min_ = 0.0
    max_ = 1.0
    x = np.random.uniform(low=24.0, high=2924.0, size=(128, 128))

    x_t_c = apply_center_width(
        x, center=CW["C"][0], width=CW["C"][1], out_range=(min_, max_)
    )
    x_t_l = apply_center_width(
        x, center=CW["L"][0], width=CW["L"][1], out_range=(min_, max_)
    )
    x_t_n = apply_center_width(
        x, center=CW["N"][0], width=CW["N"][1], out_range=(min_, max_)
    )
    assert np.min(x_t_c) >= min_
    assert np.max(x_t_c) <= max_
    assert np.min(x_t_l) >= min_
    assert np.max(x_t_l) <= max_
    assert np.min(x_t_n) >= min_
    assert np.max(x_t_n) <= max_
