from __future__ import annotations

import typing as t

from sklearn.neural_network import MLPClassifier, MLPRegressor  # type: ignore

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node


SciKitModule: t.TypeAlias = t.Union[MLPClassifier, MLPRegressor]
"""
Utility type alias for any MLP classifier or regressor implemented in Scikit-Learn.
"""


Record: t.TypeAlias = t.Dict[str, t.Any]
"""
Utility type alias for a `record` which is used for recording results.
"""

if t.TYPE_CHECKING:
    from flight.learning.types import Params

# TODO: Pair this down to real, valid datasets types for the supported ML/DL frameworks.
LearnableData: t.TypeAlias = t.Any


@t.runtime_checkable
class HasParameters(t.Protocol):
    """
    A protocol that simply requires any object to have getter and setter methods for
    the parameters of an implementing class.

    Specifically, this protocol acts as an interface for any object that could have
    trainable model parameters (e.g., `torch.nn.Module`).
    """

    def get_params(self) -> Params:
        """
        Getter method for model parameters.

        Returns:
            The trainable object's parameters.
        """

    def set_params(self, params: Params) -> None:
        """
        Setter method for model parameters.

        Args:
            params (Params): Parameters to copy into this trainable object's parameters.
        """


@t.runtime_checkable
class DataLoadable(t.Protocol):
    def train_data(self, node: Node | None = None) -> LearnableData:
        pass

    def test_data(self, node: Node | None = None) -> LearnableData | None:
        pass

    def valid_data(self, node: Node | None = None) -> LearnableData | None:
        pass
