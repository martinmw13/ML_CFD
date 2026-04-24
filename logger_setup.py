import logging
import logging.config
from enum import Enum
from typing import Final

APP_LOGGER_NAMESPACE: Final[str] = "app"

LOGGER_NAME_DATA_PREP: Final[str] = "data_prep"
LOGGER_NAME_MMM_PIPELINE: Final[str] = "mmm_pipeline"

_configured = False


class LoggingLevel(str, Enum):
    """
    Canonical logging levels used across the application configuration.

    Provides a string-based Enum aligned with Python's `logging` levels, allowing
    configuration files and environment variables to specify log verbosity using
    readable strings (e.g., "INFO", "DEBUG").
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "format": (
                "{asctime} | level={levelname} | pid={process} | ({filename}, line {lineno}) "
                'message="{message}"'
            ),
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "console_basic": {
            "format": (
                "{asctime} | level={levelname} | pid={process} | ({filename}, line {lineno}) "
                'message="{message}"'
            ),
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {},
    "handlers": {
        "console_handler": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "console_basic",
            "filters": [],
        },
    },
    # Root receives all library loggers (e.g. mmm.*); ``app.*`` propagates here too.
    # A handler only on ``app`` would miss ``logging.getLogger("mmm.training.pipeline")``.
    "root": {
        "handlers": ["console_handler"],
        "level": "INFO",
    },
    "loggers": {
        APP_LOGGER_NAMESPACE: {
            "level": "INFO",
            "propagate": True,
        },
        # Python-side TensorFlow chatter; C++ INFO lines still use TF_CPP_MIN_LOG_LEVEL.
        "tensorflow": {
            "level": "WARNING",
            "propagate": True,
        },
    },
}


def configure_logging() -> None:
    """
    Apply shared logging configuration (console handler on root, formatters).

    Safe to call from multiple entry points: only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    logging.config.dictConfig(LOGGING_CONFIG)
    _configured = True


def get_logger(component: str) -> logging.Logger:
    """
    Return a logger under the application namespace.

    Args:
        component: Short name for this module or script (e.g. ``data_prep``).

    Returns:
        A ``logging.Logger`` instance whose name is ``app.<component>``.
    """
    return logging.getLogger(f"{APP_LOGGER_NAMESPACE}.{component}")
