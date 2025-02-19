from __future__ import annotations

import typing as t

from v1.flight.learning import AbstractModule
from v1.flight.learning.params import Params
from v1.flight.learning.types import FrameworkKind

if t.TYPE_CHECKING:
    from sklearn.neural_network import MLPClassifier, MLPRegressor


class ScikitModule(AbstractModule):
    WEIGHT_KEY_PREFIX: t.Final[str] = "weight"
    BIAS_KEY_PREFIX: t.Final[str] = "bias"

    def __init__(self, module: MLPClassifier | MLPRegressor):
        self.module = module
        self._dims_initialized = False  # TODO: Check if the scikit object has this.

    ####################################################################################

    # noinspection PyMethodMayBeStatic
    @t.final
    def kind(self) -> FrameworkKind:
        return "scikit"

    def get_params(self) -> Params:
        """
        Getter method for the parameters of a trainable module (i.e., neural network)

        Returns:
            The parameters of the module.
        """
        params = []
        for i in range(self._n_layers):
            params.append((f"{self.WEIGHT_KEY_PREFIX}_{i}", self.module.coefs_[i]))
            params.append((f"{self.BIAS_KEY_PREFIX}_{i}", self.module.intercepts_[i]))
        return Params(params)

    def set_params(self, params: Params):
        """
        Setter method for the parameters of a trainable module (i.e., neural network)
        implemented in Scikit-Learn.

        Args:
            params (Params): The parameters to set.
        """
        params = params.numpy()
        param_keys = list(params.keys())
        layer_nums = set(map(lambda txt: int(txt.split("_")[-1]), param_keys))
        num_layers = max(layer_nums) + 1

        weights, biases = [], []
        for i in range(num_layers):
            w_i = params[f"{self.WEIGHT_KEY_PREFIX}_{i}"]
            b_i = params[f"{self.BIAS_KEY_PREFIX}_{i}"]
            weights.append(w_i)
            biases.append(b_i)

        self.module.coefs_ = weights
        self.module.intercepts_ = biases

    ####################################################################################

    @property
    def _n_layers(self) -> int:
        n = len(self.module.coefs_)
        if n != len(self.module.intercepts_):
            raise ValueError(
                "ScikitModule :: Inconsistent number of layers between "
                "coefficients/weights and intercepts/biases."
            )
        return n
