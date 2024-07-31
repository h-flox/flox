from __future__ import annotations

import typing as t
from collections import OrderedDict

from sklearn.neural_network import MLPClassifier, MLPRegressor

FlightDataset: t.TypeAlias = t.Any
"""
...
"""

SciKitModule: t.TypeAlias = t.Union[MLPClassifier, MLPRegressor]
"""
Utility type alias for any MLP classifier or regressor implemented in Scikit-Learn.
"""


Record: t.TypeAlias = t.Dict[str, t.Any]
"""
...
"""

RecordList: t.TypeAlias = t.List[Record]
"""
...
"""

if t.TYPE_CHECKING:
    from flight.learning.types import Params


@t.runtime_checkable
class Trainable(t.Protocol):
    def get_params(self, include_state: bool = False) -> Params:
        pass

    def set_params(self, params: Params) -> None:
        pass


class ScikitTrainable:
    WEIGHT_KEY_PREFIX = "weight"
    BIAS_KEY_PREFIX = "bias"

    def __init__(self, module: SciKitModule):
        self.module = module

    def get_params(self) -> Params:
        """

        Throws:
            - ValueError: Occurs when the `len()` of the coefficient and intercept vectors (i.e., `module.coefS_` and
              `module.intercepts_`) are not equal.

        Returns:

        """
        num_layers = len(self.module.coefs_)
        if num_layers != len(self.module.intercepts_):
            raise ValueError(
                "ScikitTrainable - Inconsistent number of layers between coefficients/weights and intercepts/biases."
            )

        params = []
        for i in range(num_layers):
            params.append((f"{self.WEIGHT_KEY_PREFIX}_{i}", self.module.coefs_[i]))
            params.append((f"{self.BIAS_KEY_PREFIX}_{i}", self.module.intercepts_[i]))

        return OrderedDict(params)

    def set_params(self, params: Params):
        param_keys = list(params.keys())
        layer_nums = map(lambda txt: int(txt.split("_")[-1]), param_keys)
        layer_nums = set(layer_nums)
        num_layers = max(layer_nums) + 1

        weights = []
        biases = []
        for i in range(num_layers):
            w_i = params[f"{self.WEIGHT_KEY_PREFIX}_{i}"]
            b_i = params[f"{self.BIAS_KEY_PREFIX}_{i}"]
            weights.append(w_i)
            biases.append(b_i)

        self.module.coefs_ = weights
        self.module.intercepts_ = biases
