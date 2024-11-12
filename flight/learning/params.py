import abc

import numpy.typing as npt


class Params(abc.ABC):
    @abc.abstractmethod
    def numpy(self) -> dict[str, npt.NDArray]:
        pass


class NpParams(Params):
    @abc.abstractmethod
    def numpy(self) -> dict[str, npt.NDArray]:
        pass


class TorchParams(Params):
    @abc.abstractmethod
    def numpy(self) -> dict[str, npt.NDArray]:
        pass
