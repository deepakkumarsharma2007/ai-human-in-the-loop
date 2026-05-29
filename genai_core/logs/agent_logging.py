import os
import logging

from logging import config as logging_config
from logging.config import dictConfig


PACKAGE_ROOT_LOGGER_NAME: str = "agent_logger"

# Logs directory for log files.
# If not set, file based log handlers are not added.
LOGS_DIR = os.environ.get("AGENT_LOGGER_LOGS_DIR", None)

# Ref: https://docs.python.org/3/library/logging.html#logging-levels
LOGS_LEVEL = os.environ.get("AGENT_LOGGER_LOGS_LEVEL", "INFO")


class DKSAgentLogger:

    
    def __init__(self) -> None:
        raise NotImplementedError("DKSAgentLogger is a static class, cannot be instantiated. Use DKSAgentLogger.get_logger() to get logger instance.")

    ###########################################################################
    # Configure logging function
    # Call this function to configure the logging.
    # Env vars can be imported here and logging configuration can be updated.
    #
    @staticmethod
    def configure_logging():
        """
        To configure logging module with dictionary.
        """
        LOGGING_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s,%(msecs)03d] [%(name)s] [%(levelname)s] [%(message)s]",
                    "datefmt": "%d %b %Y %H:%M:%S"
                },
                "detailed": {
                    "format": "[%(asctime)s,%(msecs)03d] [%(name)s] [%(levelname)s] [%(module)s] [%(lineno)d] [%(message)s]",
                    "datefmt": "%d %b %Y %H:%M:%S"
                },
                "simple": {
                    "format": "[%(levelname)s] [%(message)s]",
                },
            },
            "handlers": {
                # To handle logs on console display
                "console": {"class": "logging.StreamHandler", "formatter": "detailed"},
            },
            "loggers": {
                "agent_logger": {
                    "handlers": ["console"],
                    "level": LOGS_LEVEL,
                    "propagate": False,
                }
            },
        }

        # If logs directory set configure the required handler and loggers
        if LOGS_DIR:
            if not os.path.exists(LOGS_DIR):
                raise ValueError(
                    "Logging directory set with invalid value. Path does not exist."
                )

            # To handle logs file rotation based on time.
            LOGGING_CONFIG["handlers"]["file_timed"] = {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(LOGS_DIR, "logs.log"),
                "formatter": "detailed",
                "when": "midnight",  # Hourly rotate, other options 'S', 'M', 'H', 'D', 'W0', 'midnight'
            }
            # To handle rotation based on file size
            # This is a config example to be added in future if needed.
            LOGGING_CONFIG["loggers"]["agent_logger"]["handlers"].append(
                "file_timed"
            )

        logging_config.dictConfig(LOGGING_CONFIG)

    ###########################################################################
    #
    # Gets configured logger object from logging module.
    #
    # logger = DKSAgentLogger.get_logger() # You can segment logging like DKSAgentLogger.get_logger('core') DKSAgentLogger.get_logger('scripts') DKSAgentLogger.get_logger('security') DKSAgentLogger.get_logger('tools') DKSAgentLogger.get_logger('skills')
    #
    # logger.info('Info message')
    # logger.error('Error message')
    # logger.warning('Warning message')
    #
    @staticmethod
    def get_logger(logger_name: str = PACKAGE_ROOT_LOGGER_NAME):
        """
        Function to get logger object for agent_logger package.

    
        Returns logging module logger object to use for logging messages.
        """
        # If logger name given append it to root logger.
        if logger_name != PACKAGE_ROOT_LOGGER_NAME:
            logger_name = f"{PACKAGE_ROOT_LOGGER_NAME}.{logger_name}"

        # This always returns single logger object per logger
        logger_obj = logging.getLogger(logger_name)
        # Check if there are handlers attached to the logger, if yes, then logging already configured
        # No need to reconfigure it.
        if not logger_obj.hasHandlers():
            DKSAgentLogger.configure_logging()

        # return the logger object
        return logger_obj