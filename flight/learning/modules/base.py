from __future__ import annotations

import typing as t

from sklearn.neural_network import MLPClassifier, MLPRegressor

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
    def get_params(self) -> Params:
        pass

    def set_params(self, params: Params) -> None:
        pass


@t.runtime_checkable
class DataLoadable(t.Protocol):
    def train_data(self, node: Node | None = None) -> LearnableData:
        pass

    def test_data(self, node: Node | None = None) -> LearnableData | None:
        pass

    def valid_data(self, node: Node | None = None) -> LearnableData | None:
        pass
