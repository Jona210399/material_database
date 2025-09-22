import logging
from logging import WARNING, getLogger

LOGGER = getLogger("CifParser")
LOGGER.setLevel(WARNING)
LOGGER.propagate = False

LOGGER.handlers.clear()
LOGGER.addHandler(logging.NullHandler())

FORMATTER = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def add_file_handler(filename: str):
    remove_file_handler()

    handler = logging.FileHandler(filename)
    handler.setLevel(WARNING)
    handler.setFormatter(FORMATTER)
    LOGGER.addHandler(handler)


def remove_file_handler():
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler):
            LOGGER.removeHandler(handler)
            handler.close()


def enable_console_logging():
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler):
            return

    handler = logging.StreamHandler()
    handler.setLevel(WARNING)
    handler.setFormatter(FORMATTER)
    LOGGER.addHandler(handler)


def disable_console_logging():
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler):
            LOGGER.removeHandler(handler)
            handler.close()
