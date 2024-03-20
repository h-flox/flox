class SequentialLauncher:
    """
    A Launcher implementation that does not rely on Futures or any of the concurrent execution frameworks
    native to Python. This simply runs jobs one at a time. This is useful for debugging and confirming
    whether your defined FL process is able to run properly. Error messages using the Executors from the
    ``concurrent.futures`` module are a bit cryptic. So this is meant to alleviate this concern.
    """

    def __init__(self):
        # TODO
        raise NotImplementedError
