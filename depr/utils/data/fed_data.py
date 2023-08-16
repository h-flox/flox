from typing import Protocol

from depr.typing import WorkerID, Indices


class IndicesGenerator(Protocol):
    def sample(self) -> dict[WorkerID, Indices]:
        workers = {}
        for idx in range(num):
            n_samples = random.randint(50, 250)
            indices = random.sample(range(60_000), k=n_samples)
            workers[f"Worker-{idx}"] = worker_logic(idx=idx, indices=list(indices))
        return workers
