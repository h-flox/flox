import typing as t

import torch

FlightDataset: t.TypeAlias = t.Any

Record: t.TypeAlias = t.Dict[str, t.Any]
RecordList: t.TypeAlias = t.List[Record]
Params: t.TypeAlias = t.Mapping[str, torch.Tensor]


class Trainable(t.Protocol):
    module: FlightModule

    def get_params(self, include_buffers: bool = False) -> Params:
        pass

    def set_params(self, params: Params) -> None:
        pass

    def train(self, data: FlightDataset) -> RecordList:
        pass

    def test(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass
