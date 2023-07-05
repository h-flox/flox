import lightning as L

from flox.executor.base import BaseExec


def fit(
        executor: BaseExec,
        workers,
        aggr_logic,
        worker_logic,
        aggr_rounds: int
):
    global_module = aggr_logic.on_model_init()
    for ar in range(aggr_rounds):
        ar_workers = aggr_logic.on_worker_selection(workers)
        futures = executor.submit_jobs(workers=ar_workers, logic=worker_logic, module=global_module)
        results = [fut.result() for fut in futures]
        aggr_weights = aggr_logic(global_module, results)
        global_module.load_state_dict(aggr_weights)
        test_metrics = aggr_logic.on_model_eval(global_module)
    return global_module
