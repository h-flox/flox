import logging
import typing as t


class FlightCoordinatorLogFormat(logging.Formatter):
    green: t.Final[str] = "\x1b[32;20m"
    grey: t.Final[str] = "\x1b[38;20m"
    yellow: t.Final[str] = "\x1b[33;20m"
    red: t.Final[str] = "\x1b[31;20m"
    bold_red: t.Final[str] = "\x1b[31;1m"
    reset: t.Final[str] = "\x1b[0m"
    base_format = (
        "%(asctime)s - %(name)s - %(levelname)s"
        " - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS: t.Final[dict[int, t.Any]] = {
        logging.INFO: green + base_format + reset,
        logging.DEBUG: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger() -> logging.Logger:
    """
    Initialize the logger for the Flight framework.

    Returns:
        The initialized logger with the formatting defined by
        [`FlightCoordinatorLogFormat`][flight.logging.FlightCoordinatorLogFormat].
    """
    logger = logging.getLogger("Flight")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(FlightCoordinatorLogFormat())
    logger.addHandler(console_handler)
    return logger
