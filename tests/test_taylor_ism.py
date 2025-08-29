# test_taylor_ism.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import random_one_hot

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import SmallDeepSEA

import sys

module_path = "../tangermeme"

if module_path not in sys.path:
    sys.path.insert(0, module_path)

from taylor_ism import _predict_grad
from taylor_ism import _attribution_score
from taylor_ism import taylor_ism

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
    return random_one_hot((64, 4, 100), random_state=0).type(torch.float32)


@pytest.fixture
def X0():
    return random_one_hot((2, 4, 100), random_state=0).float()


@pytest.fixture
def alpha():
    r = numpy.random.RandomState(0)
    return torch.from_numpy(r.randn(64, 1)).type(torch.float32)


@pytest.fixture
def beta():
    r = numpy.random.RandomState(1)
    return torch.from_numpy(r.randn(64, 1)).type(torch.float32)


##


class LambdaWrapper(torch.nn.Module):
    """Wrapper that runs a given forward function instead of the default.

    Several of the classes in toy_models.py return multiple outputs but the
    attributions from taylor_ism require that there's only one output per
    example to explain. This class helps overcome the issues with having
    multiple outputs by slicing out the output we're interested in.


    Parameters
    ----------
    model: torch.nn.Module
            A PyTorch model that we want to use.

    forward: function
            A function that takes in a model and a batch of sequences and
            returns some output. Usually this is just running the forward
            function of the model and then slicing out an output.
    """

    def __init__(self, model, forward):
        super(LambdaWrapper, self).__init__()
        self.model = model
        self._forward = forward

    def forward(self, X, *args):
        return self._forward(self.model, X, *args)


def test_grad_summodel(X):
    torch.manual_seed(0)
    model = SumModel()
    y = _predict_grad(model, X, batch_size=8, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32
    assert y.sum() == X.sum()
    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
        ],
    )

    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=1, device="cpu"))
    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=64, device="cpu"))


def test_grad_flattendense(X):
    torch.manual_seed(0)
    model = FlattenDense()
    y = _predict_grad(model, X, batch_size=8, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32

    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
        ],
        4,
    )

    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=1, device="cpu"))
    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=64, device="cpu"))


def test_grad_conv(X):
    torch.manual_seed(0)
    model = Conv()
    y = _predict_grad(model, X, batch_size=8, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32

    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [0.0003, 0.0003, 0.0003, 0.0003],
                [0.0002, 0.0002, 0.0002, 0.0002],
                [-0.0004, -0.0004, -0.0004, -0.0004],
                [-0.0008, -0.0008, -0.0008, -0.0008],
            ],
            [
                [0.0003, 0.0003, 0.0003, 0.0003],
                [0.0002, 0.0002, 0.0002, 0.0002],
                [-0.0004, -0.0004, -0.0004, -0.0004],
                [-0.0008, -0.0008, -0.0008, -0.0008],
            ],
            [
                [0.0003, 0.0003, 0.0003, 0.0003],
                [0.0002, 0.0002, 0.0002, 0.0002],
                [-0.0004, -0.0004, -0.0004, -0.0004],
                [-0.0008, -0.0008, -0.0008, -0.0008],
            ],
            [
                [0.0003, 0.0003, 0.0003, 0.0003],
                [0.0002, 0.0002, 0.0002, 0.0002],
                [-0.0004, -0.0004, -0.0004, -0.0004],
                [-0.0008, -0.0008, -0.0008, -0.0008],
            ],
        ],
        4,
    )

    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=1, device="cpu"))
    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=64, device="cpu"))


def test_grad_scatter(X):
    torch.manual_seed(0)
    model = Scatter()
    y = _predict_grad(model, X, batch_size=8, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32
    assert y.sum() == X.sum() / 100

    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
            ],
            [
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
            ],
            [
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
            ],
            [
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
                [0.0025, 0.0025, 0.0025, 0.0025],
            ],
        ],
        4,
    )

    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=1, device="cpu"))
    assert_array_almost_equal(y, _predict_grad(model, X, batch_size=64, device="cpu"))


def test_grad_convdense_dense_wrapper(X):
    torch.manual_seed(0)
    model = LambdaWrapper(ConvDense(), lambda model, X: model(X)[1])
    y = _predict_grad(model, X, batch_size=2, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32

    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
            [
                [-0.0059, -0.0124, 0.0251, 0.0371],
                [-0.0080, -0.0035, -0.0416, -0.0366],
                [-0.0291, 0.0037, -0.0029, -0.0120],
                [-0.0040, -0.0147, -0.0090, -0.0027],
            ],
        ],
        4,
    )

    y = _predict_grad(model, X, batch_size=64, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32


def test_grad_convdense_conv_wrapper(X):
    torch.manual_seed(0)
    model = LambdaWrapper(ConvDense(), lambda model, X: model(X)[0])
    y = _predict_grad(model, X, batch_size=2, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32

    assert_array_almost_equal(
        y[30:34, :, 48:52],
        [
            [
                [-0.0011, -0.0011, -0.0011, -0.0011],
                [0.0018, 0.0018, 0.0018, 0.0018],
                [-0.0012, -0.0012, -0.0012, -0.0012],
                [0.0002, 0.0002, 0.0002, 0.0002],
            ],
            [
                [-0.0011, -0.0011, -0.0011, -0.0011],
                [0.0018, 0.0018, 0.0018, 0.0018],
                [-0.0012, -0.0012, -0.0012, -0.0012],
                [0.0002, 0.0002, 0.0002, 0.0002],
            ],
            [
                [-0.0011, -0.0011, -0.0011, -0.0011],
                [0.0018, 0.0018, 0.0018, 0.0018],
                [-0.0012, -0.0012, -0.0012, -0.0012],
                [0.0002, 0.0002, 0.0002, 0.0002],
            ],
            [
                [-0.0011, -0.0011, -0.0011, -0.0011],
                [0.0018, 0.0018, 0.0018, 0.0018],
                [-0.0012, -0.0012, -0.0012, -0.0012],
                [0.0002, 0.0002, 0.0002, 0.0002],
            ],
        ],
        4,
    )

    y = _predict_grad(model, X, batch_size=64, device="cpu")

    assert y.shape == (64, 4, 100)
    assert y.dtype == torch.float32


def test_grad_batch_size(X):
    torch.manual_seed(0)
    model = Scatter()
    y = _predict_grad(model, X, batch_size=68, device="cpu")
    assert y.shape == (64, 4, 100)


def test_grad_raises_shape(X):
    torch.manual_seed(0)
    model = Scatter()
    assert_raises(RuntimeError, _predict_grad, model, X[0], device="cpu")
    assert_raises(RuntimeError, _predict_grad, model, X[:, 0], device="cpu")
    assert_raises(RuntimeError, _predict_grad, model, X.unsqueeze(0), device="cpu")


def test_grad_raises_args(X, alpha, beta):
    torch.manual_seed(0)
    model = FlattenDense()
    assert_raises(
        TypeError, _predict_grad, model, X, batch_size=2, args=5, device="cpu"
    )
    assert_raises(
        AttributeError, _predict_grad, model, X, batch_size=2, args=(5,), device="cpu"
    )
    assert_raises(
        ValueError, _predict_grad, model, X, batch_size=2, args=alpha, device="cpu"
    )
    assert_raises(
        ValueError,
        _predict_grad,
        model,
        X,
        batch_size=2,
        args=(alpha[:5],),
        device="cpu",
    )
    assert_raises(
        ValueError,
        _predict_grad,
        model,
        X,
        batch_size=2,
        args=(alpha, beta[:5]),
        device="cpu",
    )


###


def test_attribution_score():
    torch.manual_seed(0)
    grads = torch.randn(64, 4, 100)

    attr = _attribution_score(grads)
    attr2 = grads - grads.mean(dim=1, keepdims=True)

    assert attr.shape == (64, 4, 100)
    assert_array_almost_equal(
        attr[30:34, :, 48:52],
        [
            [
                [-5.8536e-03, -1.3806e00, -8.4156e-01, 1.7133e00],
                [1.9718e00, -1.2490e00, -4.2352e-01, -3.4697e-01],
                [-1.1154e00, 2.3628e00, 8.1413e-01, -1.8472e-01],
                [-8.5058e-01, 2.6679e-01, 4.5095e-01, -1.1816e00],
            ],
            [
                [-8.5315e-01, -5.7350e-01, 1.7855e-01, -6.5356e-01],
                [5.3122e-01, -1.1553e00, 1.8099e-01, 1.0569e00],
                [-1.7029e00, 9.5211e-01, -4.7820e-01, -1.7130e-01],
                [2.0249e00, 7.7671e-01, 1.1867e-01, -2.3202e-01],
            ],
            [
                [6.7352e-01, 6.8433e-01, 9.6243e-02, -7.1465e-01],
                [-6.7257e-01, -8.7914e-01, -9.7078e-01, 9.5695e-01],
                [-4.3117e-01, -1.4509e00, -3.2867e-01, 1.2065e00],
                [4.3022e-01, 1.6457e00, 1.2032e00, -1.4488e00],
            ],
            [
                [-1.0343e00, -6.0057e-01, -1.3251e00, 2.8819e00],
                [3.4446e-01, -2.4408e-02, -4.5240e-01, -7.7934e-01],
                [1.7264e00, -4.9770e-04, -1.2533e00, -1.1771e00],
                [-1.0365e00, 6.2548e-01, 3.0307e00, -9.2548e-01],
            ],
        ],
        4,
    )
    assert_array_almost_equal(attr, attr2, 4)


###


def test_taylor_ism(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = taylor_ism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    assert_array_almost_equal(
        X_attr[:, :, :4],
        [
            [
                [-0.0003, -0.0000, 0.0000, 0.0005],
                [0.0000, 0.0000, -0.0005, -0.0000],
                [0.0000, -0.0000, -0.0000, -0.0000],
                [-0.0000, 0.0021, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.0012, 0.0000],
                [0.0013, -0.0000, -0.0000, -0.0000],
                [0.0000, 0.0020, -0.0000, 0.0000],
                [-0.0000, -0.0000, -0.0000, -0.0008],
            ],
        ],
        4,
    )


def test_taylor_ism_hypothetical(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = taylor_ism(model, X0, hypothetical=True, device="cpu")
    X_attr2 = taylor_ism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    assert_array_almost_equal(
        X_attr[:, :, :4],
        [
            [
                [-3.4767e-04, -1.0026e-03, 1.1671e-03, 4.7212e-04],
                [7.6799e-04, 3.0037e-05, -5.0321e-04, -1.9246e-03],
                [4.4439e-04, -1.1574e-03, -1.7747e-03, -5.4200e-05],
                [-8.6472e-04, 2.1300e-03, 1.1108e-03, 1.5067e-03],
            ],
            [
                [1.1239e-05, 4.3108e-04, 1.2186e-03, 5.1815e-04],
                [1.2906e-03, -1.2843e-03, -2.9760e-04, -8.0590e-04],
                [1.2349e-04, 2.0470e-03, -8.4870e-05, 1.0950e-03],
                [-1.4254e-03, -1.1938e-03, -8.3611e-04, -8.0722e-04],
            ],
        ],
        4,
    )
    assert_array_almost_equal(X_attr * X0, X_attr2, 4)


def test_taylor_ism_ordering(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    X_attr = taylor_ism(model, X0, device="cpu")

    assert X_attr.shape == (2, 4, 100)
    assert X_attr.dtype == torch.float32

    X_attr2 = taylor_ism(model, X0[1:], device="cpu")
    assert_array_almost_equal(X_attr[1:, :, :], X_attr2, 2)


def test_taylor_ism_target(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(3)
    X_attr = taylor_ism(model, X0, device="cpu")
    X_attr0 = taylor_ism(model, X0, target=0, device="cpu")
    X_attr1 = taylor_ism(model, X0, target=1, device="cpu")
    X_attr2 = taylor_ism(model, X0, target=2, device="cpu")

    assert X_attr0.shape == (2, 4, 100)
    assert X_attr0.dtype == torch.float32

    assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
    assert_array_almost_equal((X_attr0 + X_attr1 + X_attr2) / 3, X_attr)


def test_taylor_ism_raw_output(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)
    grads = taylor_ism(model, X0, raw_outputs=True, device="cpu")

    assert grads.shape == (2, 4, 100)
    assert grads.dtype == torch.float32

    assert_array_almost_equal(
        grads[:, :, :4],
        [
            [
                [-7.7278e-04, -8.5049e-04, 7.7635e-04, 1.6561e-03],
                [3.4288e-04, 1.8218e-04, -8.9400e-04, -7.4064e-04],
                [1.9277e-05, -1.0053e-03, -2.1655e-03, 1.1298e-03],
                [-1.2898e-03, 2.2822e-03, 7.2000e-04, 2.6907e-03],
            ],
            [
                [3.6527e-04, -5.4919e-04, 1.6762e-03, -3.4353e-04],
                [1.6447e-03, -2.2645e-03, 1.6002e-04, -1.6676e-03],
                [4.7752e-04, 1.0667e-03, 3.7275e-04, 2.3328e-04],
                [-1.0713e-03, -2.1741e-03, -3.7849e-04, -1.6689e-03],
            ],
        ],
        4,
    )


def test_taylor_ism_equivalence(X0):
    torch.manual_seed(0)
    model = SmallDeepSEA(5)

    X_attr = taylor_ism(model, X0, hypothetical=True, device="cpu")
    grads = taylor_ism(model, X0, raw_outputs=True, device="cpu")

    attr = _attribution_score(grads)
    assert_array_almost_equal(X_attr, attr, 4)


def test_taylor_ism_sum_model(X0):
    model = SumModel()
    grads = taylor_ism(model, X0, raw_outputs=True, device="cpu")
    assert grads.shape == (2, 4, 100)
    assert grads.dtype == torch.float32

    assert_array_almost_equal(
        grads[:, :, :4],
        [
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.2500, 0.2500],
            ],
        ],
    )
