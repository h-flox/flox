import inspect
from enum import Flag, auto
from functools import wraps


class CoordinatorState:
    ...


class CoordinatorEvents(Flag):
    BEFORE_INITIALIZE_MODEL = auto()
    AFTER_INITIALIZE_MODEL = auto()

    BEFORE_INITIALIZE_REMOTE_ENGINE = auto()
    AFTER_INITIALIZE_REMOTE_ENGINE = auto()

    BEFORE_WORKER_SELECTION = auto()
    WORKER_SELECTION = auto()
    AFTER_WORKER_SELECTION = auto()

    BEFORE_SHARE_GLOBAL_MODEL = auto()
    BEFORE_SUBMIT_AGGR_JOBS = auto()
    BEFORE_SUBMIT_WORKER_JOBS = auto()

    AFTER_CHILDREN_RESPONSES = auto()


def on(event_type: CoordinatorEvents):
    def decorator(func):
        setattr(func, "_event_type", event_type)  # Store metadata

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
