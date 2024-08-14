from __future__ import annotations

from collections import OrderedDict

from flight.learning.modules.prototypes import SciKitModule
from flight.learning.types import Params


class ScikitTrainable:
    WEIGHT_KEY_PREFIX = "weight"
    BIAS_KEY_PREFIX = "bias"

    def __init__(self, module: SciKitModule):
        self.module = module

    def get_params(self) -> Params:
        """

        Throws:
            - ValueError: Occurs when the `len()` of the coefficient and intercept
              vectors (i.e., `module.coefS_` and `module.intercepts_`) are not equal.

        Returns:

        """
        num_layers = len(self.module.coefs_)
        if num_layers != len(self.module.intercepts_):
            raise ValueError(
                "ScikitTrainable - Inconsistent number of layers between "
                "coefficients/weights and intercepts/biases."
            )

        params = []
        for i in range(num_layers):
            params.append((f"{self.WEIGHT_KEY_PREFIX}_{i}", self.module.coefs_[i]))
            params.append((f"{self.BIAS_KEY_PREFIX}_{i}", self.module.intercepts_[i]))

        return OrderedDict(params)

    def set_params(self, params: Params):
        param_keys = list(params.keys())
        layer_nums = set(map(lambda txt: int(txt.split("_")[-1]), param_keys))
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
