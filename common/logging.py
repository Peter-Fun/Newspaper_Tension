# Generic Packages
import os
import sys
import atexit
import logging
import functools


def setup_logging(output_dir=None):
    """
    This method sets up the root logger which will print to the stdout (and output_dir/stdout.log if specified)
    """

    # Specify the logging format.
    _FORMAT = "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)3d] %(message)s"

    # Set up the logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Prevent logger"s ancestors from obtaining log msg as well.
    logger.propagate = False
    plain_formatter = logging.Formatter(_FORMAT, datefmt="%m/%d %H:%M:%S",)

    # Set up logging to stdout
    out_handler = logging.StreamHandler(stream=sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(plain_formatter)
    logger.addHandler(out_handler)

    # Set up logging to file
    if output_dir is not None:
        fname = os.path.join(output_dir, "stdout.log")
        file_handler = logging.StreamHandler(_cached_log_stream(filename=fname))
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)


def getlogger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy. If a name is given, the named
    logger will automatically be a child of the root logger.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """
    return a cached "io" variable if the function was previously called.
    """
    io = open(filename, "a+", buffering=1024)
    atexit.register(io.close)
    return io