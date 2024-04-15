from enum import Enum
import logging as log
import sys
import traceback
from functools import wraps
import os
import datetime
from datetime import datetime as dt


class LogLevel(Enum):
    """
    Simple Helper Enum for Logging Levels. Will assist with ensuring consistency.
    """
    L_NOTSET = log.NOTSET
    L_DEBUG = log.DEBUG
    L_INFO = log.INFO
    L_WARNING = log.WARNING
    L_ERROR = log.ERROR
    L_CRITICAL = log.CRITICAL


def log_it(level: LogLevel=LogLevel.L_INFO, other: str="{0}", print_results=True, loggerName="db"):
    def wrapper_log_it(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = log.getLogger(loggerName)
            try:
                logger.debug(f"Function being triggered: {func.__name__}")
                f = func(*args, **kwargs)
                strf = ""
                if type(f) is tuple or type(f) is list:
                    strf = ' :: '.join([str(e) for e in f])
                else:
                    strf = str(f)
                if print_results:
                    logger.log(level.value, other.format(strf))
                else:
                    logger.log(level.value, f"Function Call Complete. (Anonymous Response) '{print_results=}'")
                return f
            except BaseException as BE:
                exc = sys.exc_info()
                tr = traceback.format_exc()
                logger.error(f"Error while handling {func.__name__}: {str(exc[0].__name__)}\n\t{tr}")
                print(f"Error while handling {func.__name__}: {str(exc[0].__name__)}\n\t{tr}")
                return (False, f"[ERROR] Type:{exc[0].__name__}, Message: {BE}")
        return wrapper
    return wrapper_log_it


# Logging Stuff
def make_logging(DB_LOG_DIR_PATH, DB_LOG_LEVEL, DB_LOG_FORMAT):
    """
    Run this method first to instantiate a copy of the logger. Note: does not return said module. Instead, this just creates a config (logger file) and uses root logger.
    :return:
    """
    os.makedirs(DB_LOG_DIR_PATH, exist_ok=True)
    dtstr: str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    fname: str = dtstr + "_db.log"
    fpath = os.path.join(DB_LOG_DIR_PATH, fname)
    log.basicConfig(filename=f"{fpath}", level=DB_LOG_LEVEL.value, format=DB_LOG_FORMAT)
    del fpath, fname, dtstr
    logger = log.getLogger()
    print("Logging Online!")
    logger.info("Logger Online!")
    del logger


def log_line(c, loggername="methods", level: LogLevel = LogLevel.L_INFO, other="{0}"):
    logger = log.getLogger(loggername)
    strf = ""
    if type(c) is tuple or type(c) is list:
        strf = ' :: '.join([str(e) for e in c])
    else:
        strf = str(c)
    logger.log(level.value, other.format(strf))