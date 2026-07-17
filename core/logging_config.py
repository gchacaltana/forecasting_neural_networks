#!/usr/bin/env python3
"""
Explicit logging configuration for the forecasting application.

Use ``logging`` for errors and diagnostics; keep interactive UI messages on
``Console``.
"""
from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with an explicit format.

    Args:
        level: Minimum log level for the root logger.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)
