import typing as t

FlightDataset: t.TypeAlias = t.Any
FlightModule: t.TypeAlias = t.Any

Record: t.TypeAlias = t.Dict[str, t.Any]
RecordList: t.TypeAlias = t.List[Record]


class Trainable(t.Protocol):
    module: FlightModule

    def get_params(self, include_buffers: bool = False):
        pass

    def set_params(self):
        pass

    def train(self, data: FlightDataset) -> RecordList:
        pass

    def test(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass
