import lightning as L

from torch.utils.data import Dataset, DataLoader
from typing import Any, Protocol, runtime_checkable, TypeVar

T_co = TypeVar('T_co', covariant=True)


@runtime_checkable
class WorkerLogicInterface(Protocol):
    idx: Any

    def on_module_recv(self):
        ...

    def on_data_fetch(self) -> Dataset[T_co]:
        ...

    def on_module_fit(
            self,
            module: L.LightningModule,
            dataloader: DataLoader
    ) -> L.LightningModule:
        ...

    def on_module_send(self) -> dict[str, Any]:
        ...

    def __len__(self) -> int:
        ...


class SimpleWorkerLogic:
    def __init__(self, idx: Any, **kwargs):
        self.idx = idx

    def on_model_recv(self):
        pass

    def on_data_fetch(self):
        pass

    def on_module_fit(
            self,
            module: L.LightningModule,
            dataloader: DataLoader
    ) -> dict[str, Any]:
        import warnings
        from lightning import Trainer
        from lightning_utilities.core.rank_zero import log as device_logger
        device_logger.disabled = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=3,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                logger=False
            )
            trainer.fit(module, dataloader)

        return {
            "worker_id": self.idx,
            "module": module,
            "metrics": trainer.callback_metrics
        }

    def on_model_send(self) -> dict[str, Any]:
        pass

    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"Worker({self.idx})"
