from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from depr.worker import WorkerLogicInterface


def launch_aggregation_task():
    pass


def launch_local_fitting_task(
    logic: WorkerLogicInterface, module: L.LightningModule, batch_size: int = 32
) -> tuple[Any, L.LightningModule]:
    data_loader = DataLoader(logic.on_data_fetch(), batch_size=batch_size, shuffle=True)
    res = logic.on_module_fit(module, data_loader)
    return logic.idx, res
