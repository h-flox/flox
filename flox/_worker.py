import lightning as L

from typing import Any


class WorkerLogic:
    def __init__(self, idx: Any):
        self.idx = idx

    def on_model_recv(self):
        pass

    def on_data_fetch(self):
        pass

    def on_module_fit(self, module: L.LightningModule, data_loader):
        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3
        )
        trainer.fit(module, data_loader)
        return module

    def on_model_send(self) -> dict[str, Any]:
        pass

    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"Worker({self.idx})"
