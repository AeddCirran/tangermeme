# test_saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

import sys
import os

module_path = "../tests"

if module_path not in sys.path:
    sys.path.insert(0, module_path)

from toy_models import SumModel
from toy_models import FlattenDense
from toy_models import Conv
from toy_models import Scatter
from toy_models import ConvDense
from toy_models import SmallDeepSEA

module_path = "../tangermeme"

if module_path not in sys.path:
    sys.path.insert(0, module_path)

from tism import _edit_distance_one
from tism import _attribution_score
from tism import tism

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
    return random_one_hot((2, 4, 10), random_state=0)


@pytest.fixture
def X0():
    return random_one_hot((2, 4, 100), random_state=0).float()


###


def test_edit_distance_one(X):
    X_one = _edit_distance_one(X[0], 0, -1)

    assert X_one.dtype == torch.int8
    assert X_one.shape == (10, 4, 10)
    assert X_one.sum() == 90

    assert_array_almost_equal(
        X_one[:4],
        [
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
        ],
    )

    assert_array_almost_equal(
        X_one[-4:],
        [
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
        ],
    )


def test_edit_distance_one_start_end(X):
    X_one = _edit_distance_one(X[0], 2, 5)
    assert X_one.shape == (3, 4, 10)

    assert_array_almost_equal(
        X_one,
        [
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
            ],
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            ],
        ],
    )


###


def test_attribution_score():
    torch.manual_seed(0)
    y_hat = torch.randn(1, 4, 10, 1)

    attr = _attribution_score(y_hat)
    attr2 = torch.mean(y_hat, dim=1)
    attr2 -= torch.sum(attr2, dim=1, keepdims=True)

    assert attr.shape == (1, 10, 1)
    assert_array_almost_equal(
        attr,
        [
            [
                [0.9953],
                [0.3274],
                [0.9878],
                [1.4904],
                [0.9424],
                [0.7294],
                [0.3990],
                [-0.1595],
                [0.5885],
                [0.9445],
            ]
        ],
        4,
    )
    assert_array_almost_equal(attr, attr2, 4)


###


def test_tism(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = tism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    assert_array_almost_equal(
        X_attr[:, :, :3],
        [
            [
                [9.3128e-04, -0.0000e00, 0.0000e00],
                [0.0000e00, -0.0000e00, 7.5962e-04],
                [0.0000e00, -0.0000e00, -0.0000e00],
                [0.0000e00, 1.7185e-03, 0.0000e00],
            ],
            [
                [-0.0000e00, 0.0000e00, -1.9119e-04],
                [3.3933e-04, 0.0000e00, -0.0000e00],
                [-0.0000e00, 4.8020e-03, -0.0000e00],
                [-0.0000e00, 0.0000e00, -0.0000e00],
            ],
        ],
        4,
    )


def test_tism_hypothetical(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = tism(model, X0, hypothetical=True, device="cpu")
    X_attr2 = tism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    assert_array_almost_equal(
        X_attr[:, :, :3],
        [
            [
                [0.0009, -0.0012, 0.0024],
                [0.0018, -0.0005, 0.0008],
                [0.0015, -0.0015, -0.0003],
                [0.0003, 0.0017, 0.0021],
            ],
            [
                [-0.0008, 0.0031, -0.0002],
                [0.0003, 0.0016, -0.0017],
                [-0.0008, 0.0048, -0.0015],
                [-0.0022, 0.0016, -0.0022],
            ],
        ],
        4,
    )
    assert_array_almost_equal(X_attr * X0, X_attr2, 4)


def test_tism_start_end(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = tism(model, X0, start=50, end=60, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    assert_array_almost_equal(
        X_attr[:, :, :3],
        [
            [
                [0.0008, -0.0000, 0.0000],
                [0.0000, -0.0000, 0.0007],
                [0.0000, -0.0000, -0.0000],
                [0.0000, 0.0014, 0.0000],
            ],
            [
                [-0.0000, 0.0000, -0.0002],
                [0.0002, 0.0000, -0.0000],
                [-0.0000, 0.0050, -0.0000],
                [-0.0000, 0.0000, -0.0000],
            ],
        ],
        4,
    )


def test_tism_start_end_hypothetical(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = tism(model, X0, start=50, end=60, hypothetical=True, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32
    assert_array_almost_equal(
        X_attr[:, :, :3],
        [
            [
                [0.0008, -0.0011, 0.0020],
                [0.0017, -0.0008, 0.0007],
                [0.0015, -0.0018, -0.0005],
                [0.0005, 0.0014, 0.0017],
            ],
            [
                [-0.0011, 0.0034, -0.0002],
                [0.0002, 0.0017, -0.0017],
                [-0.0009, 0.0050, -0.0015],
                [-0.0025, 0.0017, -0.0022],
            ],
        ],
        4,
    )


def test_tism_ordering(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = tism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    X_attr2 = tism(model, X0[1:], device="cpu")
    assert_array_almost_equal(X_attr[1:, :, :], X_attr2, 2)


def test_tism_raw_output(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    y_hat = tism(model, X0, raw_outputs=True, device="cpu")

    assert y_hat.shape == (2, 100, 4, 100)
    test_hat = y_hat[:, :3, :, :4]
    assert_array_almost_equal(
        y_hat[:, :3, :, :4],
        [
            [
                [
                    [6.7759e-04, -1.0659e-03, -6.3947e-04, -3.7797e-04],
                    [2.1587e-05, -1.3383e-03, -1.6568e-03, -1.3157e-03],
                    [4.1197e-04, -2.9205e-03, -3.2084e-03, 2.8609e-03],
                    [-5.5471e-04, 1.8724e-03, -9.9305e-04, 2.8035e-03],
                ],
                [
                    [8.7662e-05, -2.7463e-03, 7.5088e-04, 8.5433e-04],
                    [9.0302e-04, 8.6148e-05, -2.7063e-03, -1.1085e-03],
                    [-9.4512e-05, 5.5218e-04, -3.0460e-03, 2.2100e-03],
                    [-1.2501e-03, 7.0730e-04, -2.9267e-04, 2.1584e-03],
                ],
                [
                    [4.6398e-04, -2.0577e-03, -6.2213e-04, 4.4780e-05],
                    [5.3686e-04, -1.1429e-03, -1.7124e-03, -5.2875e-05],
                    [-8.9237e-06, -4.8048e-04, -7.1327e-04, 1.3974e-03],
                    [-1.5498e-03, 1.8308e-03, -3.8710e-04, 2.6244e-05],
                ],
            ],
            [
                [
                    [4.0065e-04, -1.6698e-03, 2.1764e-03, -7.2418e-04],
                    [-2.4576e-05, -1.1897e-03, -3.0716e-04, -1.9316e-03],
                    [-9.5836e-05, 8.6613e-04, 4.5646e-04, -5.9736e-04],
                    [-5.9764e-04, -8.5693e-04, -1.7029e-04, -1.2249e-03],
                ],
                [
                    [-9.6595e-05, -1.5032e-03, 2.3306e-03, 4.1300e-04],
                    [-1.0076e-03, -2.8745e-04, 6.8927e-04, -1.9026e-03],
                    [-8.1081e-04, 1.2700e-03, 1.0046e-03, 8.4623e-04],
                    [-1.1881e-03, -5.4524e-04, -7.6465e-04, -7.0240e-04],
                ],
                [
                    [2.3968e-04, 3.6893e-04, 1.0067e-04, -1.0821e-03],
                    [-1.3690e-04, -1.8389e-03, 2.7660e-04, -1.4171e-03],
                    [-1.2796e-03, 2.6171e-03, -1.9052e-04, 1.1677e-03],
                    [-3.4915e-04, -3.1890e-03, 4.1846e-04, -1.9074e-03],
                ],
            ],
        ],
        4,
    )


def test_tism_equivalence(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)

    X_attr = tism(model, X0, hypothetical=True, device="cpu")
    y_hat = tism(model, X0, raw_outputs=True, device="cpu")

    attr = _attribution_score(y_hat)
    assert_array_almost_equal(X_attr, attr, 4)


def test_tism_sum_model(X):
    model = SumModel()
    y_hat = tism(model, X, raw_outputs=True, device="cpu")

    assert y_hat.shape == (2, 10, 4, 10)

    assert_array_almost_equal(
        y_hat[:, :3, :, :3],
        [
            [
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
            ],
            [
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
                [
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.2500],
                ],
            ],
        ],
    )
