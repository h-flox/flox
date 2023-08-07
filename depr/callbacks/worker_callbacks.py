from typing import Protocol


class WorkerCallbacks(Protocol):
    def on_model_recv(self):
        ...

    def on_data_fetch(self):
        ...

    def on_module_fit(self):
        ...

    def on_module_send(self):
        ...
