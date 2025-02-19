from v1.flight.learning import AbstractModule
from v1.flight.learning.scikit import ScikitModule
from v1.flight.learning.torch import TorchModule


def _test_scikit_global_module():
    pass


def _test_torch_global_module():
    pass


def test_global_module(module: AbstractModule):
    if isinstance(module, TorchModule):
        _test_torch_global_module()
    elif isinstance(module, ScikitModule):
        _test_scikit_global_module()
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
