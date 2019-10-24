import logging
import os

try:
    from StringIO import StringIO as StringBuffer
except ImportError:
    from io import StringIO as StringBuffer

import logging

from .helper import get_git


class DLogger:
    @staticmethod
    def logger():
        return logging.getLogger("DeepRL")

    @staticmethod
    def get_string_logger():
        log_capture_string = StringBuffer()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        DLogger.logger().addHandler(ch)
        return log_capture_string

    @staticmethod
    def remove_handlers():
        DLogger.logger().handlers = []

    @staticmethod
    def set_up_logger():
        # create logger with 'spam_application'
        logger = logging.getLogger("DeepRL")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        # fh = logging.FileHandler('spam.log')
        # fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        # ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        # logger.addHandler(fh)
        logger.addHandler(ch)

    @staticmethod
    def add_loggingfile(path, file):
        if not os.path.exists(path):
            os.makedirs(path)

        fh = logging.FileHandler(path + file)
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        DLogger.logger().addHandler(fh)
        return fh

    @staticmethod
    def remove_handler(hndlr):
        DLogger.logger().removeHandler(hndlr)


class LogFile:
    def __init__(self, path, file):
        self.h = DLogger.add_loggingfile(path, file)

    def __enter__(self):
        DLogger.logger().debug("version control: " + str(get_git()))

    def __exit__(self, type, value, traceback):
        self.h.close()
        DLogger.remove_handler(self.h)
