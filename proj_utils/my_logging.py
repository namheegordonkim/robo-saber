import logging
import os
import sys


def get_logger(run_name: str, out_name: str, logdir: str):
    # Create a logger object
    logger = logging.getLogger(f"{run_name}][{out_name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create a file handler to write log messages to a file
    log_path = os.path.abspath(os.path.join(logdir, f"log_{out_name}.txt"))
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.ERROR)

    # Create a formatter for the log messages
    fmt = "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(err_handler)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
