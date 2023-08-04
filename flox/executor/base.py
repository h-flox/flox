from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def submit(self):
        pass

    @abstractmethod
    def wait(self):
        pass
