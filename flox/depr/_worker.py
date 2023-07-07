import lightning as L

from typing import Any


class WorkerLogic:
    def __init__(self, idx: Any, **kwargs):
        self.idx = idx

    def on_model_recv(self):
        pass

    def on_data_fetch(self):
        pass

    def on_module_fit(self, module: L.LightningModule, dataloader):
        from lightning import Trainer
        trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False
        )
        trainer.fit(module, dataloader)
        return module

    def on_model_send(self) -> dict[str, Any]:
        pass

    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"Worker({self.idx})"
