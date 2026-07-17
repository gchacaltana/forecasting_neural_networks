#!/usr/bin/env python3
"""
Console helpers for the interactive forecasting CLI.

Provides timestamped output, highlighted section headers, and pause prompts
used throughout the training and prediction workflow.
"""
from __future__ import annotations

import os
from datetime import datetime
from logging import Logger

import pytz

from core.constants import (
    COLOR_GREEN,
    COLOR_YELLOW,
    END_COLOR,
    HIGHLIGHT_SEPARATOR_LEN,
    TIMEZONE_NAME,
)

timezone = pytz.timezone(TIMEZONE_NAME)


class Console:
    """Static helpers for interactive console output."""

    @staticmethod
    def outline(content: str, logging: Logger | None = None) -> None:
        """Print a timestamped message and optionally forward it to a logger.

        Args:
            content: Message to display.
            logging: Optional logger that receives the same message via ``info``.
        """
        now_utc = datetime.now()
        now = now_utc.astimezone(timezone)
        outline = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {content}"
        print(outline)
        if logging:
            logging.info(outline)

    @staticmethod
    def highlight(message: str) -> None:
        """Clear the screen and print a highlighted section header.

        Args:
            message: Header text to display.
        """
        os.system("clear")
        print(f"\n{COLOR_YELLOW}{message}{END_COLOR}")
        print(f"\n{COLOR_YELLOW}{'*' * HIGHLIGHT_SEPARATOR_LEN}{END_COLOR}\n")

    @staticmethod
    def stop_continue(message: str) -> None:
        """Pause execution until the user presses Enter.

        Args:
            message: Prompt shown while waiting.
        """
        input(f"\n{COLOR_GREEN}{message}{END_COLOR}\n")
