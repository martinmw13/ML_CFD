"""Centralized logging configuration for the mlcfd package."""

from __future__ import annotations

import logging
import logging.config
from enum import Enum
from typing import Final

MLCFD_LOGGER_NAMESPACE: Final[str] = "mlcfd"

LOGGER_NAME_CLI: Final[str] = "cli"
LOGGER_NAME_IO: Final[str] = "io"
LOGGER_NAME_MODELS: Final[str] = "models"
LOGGER_NAME_PIPELINE: Final[str] = "pipeline"
LOGGER_NAME_PREPROCESSING: Final[str] = "preprocessing"
LOGGER_NAME_VISUALIZATION: Final[str] = "visualization"

_configured = False


class LoggingLevel(str, Enum):
    """String levels aligned with the standard logging module."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"


LOGGING_CONFIG: dict[str, object] = {
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
    "root": {
        "handlers": ["console_handler"],
        "level": "INFO",
    },
    "loggers": {
        MLCFD_LOGGER_NAMESPACE: {
            "level": "INFO",
            "propagate": True,
        },
        "matplotlib": {
            "level": "WARNING",
            "propagate": True,
        },
        "sklearn": {
            "level": "WARNING",
            "propagate": True,
        },
    },
}


def configure_logging(level: LoggingLevel | str | None = None) -> None:
    """Apply shared logging configuration (console handler on root, formatters).

    Safe to call from multiple entry points: only the first call takes effect.
    Subsequent calls are ignored unless logging is reconfigured externally.

    Args:
        level: Optional override for root and ``mlcfd`` logger levels.
    """
    global _configured
    if _configured:
        return
    if level is not None:
        level_name = level.value if isinstance(level, LoggingLevel) else str(level)
        root = LOGGING_CONFIG["root"]
        if isinstance(root, dict):
            root["level"] = level_name
        mlcfd_logger = LOGGING_CONFIG["loggers"]
        if isinstance(mlcfd_logger, dict):
            entry = mlcfd_logger.get(MLCFD_LOGGER_NAMESPACE)
            if isinstance(entry, dict):
                entry["level"] = level_name
    logging.config.dictConfig(LOGGING_CONFIG)
    _configured = True


def get_logger(component: str) -> logging.Logger:
    """Return a logger under the ``mlcfd`` namespace.

    Args:
        component: Short name for the subsystem (for example, ``"pipeline"``).

    Returns:
        Logger named ``mlcfd.<component>``.
    """
    return logging.getLogger(f"{MLCFD_LOGGER_NAMESPACE}.{component}")
