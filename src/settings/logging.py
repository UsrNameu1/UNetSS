import sys

from loguru import logger

from settings import log_file_path


class StreamToLogger:
    """
    capture stdout byte buffer and redirect it to file output
    """

    def __init__(self, log_level="DEBUG"):
        self.log_level = log_level

    def write(self, buf: bytes):
        for line in buf.rstrip().splitlines():
            logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def configure_logging():
    """
    configure logging settings
    """
    logger.add(log_file_path, rotation="500 MB")
    sys.stdout = StreamToLogger()
    sys.stderr = StreamToLogger("ERROR")
