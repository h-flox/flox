import typing as t

import numpy as np
import pytest
import torch

from flight.learning.parameters import (
    NumpyParams,
    Params,
    ParamKinds,
    parameters,
    TorchParams,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def weights_np(rng) -> np.ndarray:
    return rng.uniform(low=0, high=100, size=(5,))


@pytest.fixture
def weights_torch(weights_np) -> torch.Tensor:
    return torch.from_numpy(weights_np)


@pytest.fixture
def weights_list(weights_np) -> list[float]:
    return weights_np.tolist()


def test_parameters_fn(weights_np, weights_torch, weights_list):
    def _test(weight_data: t.Any, inferred_kind_for_auto_backend: ParamKinds):
        # Format the data as both a list of tuples, a standard dictionary, amd
        # a generator of tuples. This is to test the `parameters` function
        # with different input formats for the underlying `c.OrderedDict`.
        #
        # NOTE: For the generator (`pgenr`), we use a lambda function to create
        # the generator since it is used multiple times in the loop below, which
        # means it gets "used up" and is left empty on the successive iterations.
        # The lambda function ensures it's recreated each time we call it.
        plist = [(f"l{i}", weight_data) for i in range(10)]
        pdict = {f"l{i}": weight_data for i in range(10)}
        pgenr = lambda: ((f"l{i}", weight_data) for i in range(10))

        # Answer sheet for expected kinds based on the backend.
        backend_answers = {
            "auto": inferred_kind_for_auto_backend,
            "numpy": ParamKinds.NUMPY,
            "torch": ParamKinds.TORCH,
        }

        for backend, ans in backend_answers.items():
            for _p in [pdict, plist, pgenr()]:
                params = parameters(_p, backend=backend)

                # Check the inferred kind matches the expected answer
                assert params.kind == ans
                assert isinstance(params, Params)
                assert isinstance(params.numpy(), NumpyParams)
                assert isinstance(params.torch(), TorchParams)

    _test(weights_list, ParamKinds.NUMPY)
    _test(weights_np, ParamKinds.NUMPY)
    _test(weights_torch, ParamKinds.TORCH)


# def test_validate_torch_params(param_data):
#     weights = torch.tensor(param_data)
#     params = {f"p{i}": weights for i in range(10)}
#     assert infer_param_kind(weights) == ParamKinds.TORCH
#     assert validate_param_kind(params) == ParamKinds.TORCH
#
#
# def test_params_cls(param_data):
#     weights = np.array(param_data)
#     params = Params({f"p{i}": weights for i in range(10)})
#
#     params_torch = params.torch()
#     params_numpy = params.numpy()
#
#     assert isinstance(params_numpy, dict)
#     assert isinstance(params_torch, dict)
#
#     assert isinstance(params_numpy, c.OrderedDict)
#     assert isinstance(params_torch, c.OrderedDict)
#
#     for key in params:
#         assert isinstance(params_numpy[key], np.ndarray)
#         assert isinstance(params_torch[key], torch.Tensor)
#         assert float(params_numpy[key][0] == float(params_torch[key][0]))
#
#
# def test_params_create_method():
#     rng = np.random.default_rng(42)
#     for i in [1, 10, 100]:
#         for j in [1, 2, 3, 4, 5]:
#             for k in [1, 2, 3]:
#                 p = Params.create(
#                     [
#                         ("a", rng.uniform(size=(i, j, k))),
#                         ("b", rng.uniform(size=(i, j, k))),
#                         ("c", rng.uniform(size=(i, j, k))),
#                     ]
#                 )
#                 assert p.numpy().inferred_kind == ParamKinds.NUMPY
#                 assert p.torch().inferred_kind == ParamKinds.TORCH
#
#
# def test_validate_numpy_params(param_data):
#     weights = np.array(param_data)
#     params = {f"p{i}": weights for i in range(10)}
#     assert parameters(params, backend="numpy").kind == ParamKinds.NUMPY
#     assert parameters(params, backend="torch") == ParamKinds.NUMPY
