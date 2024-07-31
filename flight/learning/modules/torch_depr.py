from __future__ import annotations

import typing as t
from collections import OrderedDict

if t.TYPE_CHECKING:
    from flight.learning.types import Params


class TorchTrainableMixins:
    def get_params(self, include_state: bool = False) -> Params:
        if include_state:
            params = {name: value.clone() for name, value in self.state_dict().items()}
        else:
            params = {
                name: param.data.clone() for name, param in self.named_parameters()
            }
        return OrderedDict(params)

    def set_params(self, params: Params) -> None:
        state_dict = self.state_dict()
        for name in state_dict:
            state_dict[name] = params[name].clone()
