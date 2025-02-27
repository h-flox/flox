class WorkerState:
    pass


def worker_job():
    from ignite.engine import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset

    from flight.events import WorkerEvents
    from flight.learning.module import TorchDataModule, TorchModule

    ####################################################################

    WorkerEvents.STARTED

    ####################################################################

    ...  # TODO

    ####################################################################

    WorkerEvents.COMPLETED

    return None
