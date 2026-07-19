#!/usr/bin/env python3
"""
Console helpers for the interactive forecasting CLI.
"""
from __future__ import annotations

import os
from datetime import datetime
from logging import Logger
from typing import Any

import pytz

from core.constants import (
    COLOR_GREEN,
    COLOR_YELLOW,
    END_COLOR,
    HIGHLIGHT_SEPARATOR_LEN,
    TIMEZONE_NAME,
)

timezone = pytz.timezone(TIMEZONE_NAME)


def clear_screen() -> None:
    """Clear the terminal screen on Windows (``cls``) or Unix (``clear``)."""
    os.system("cls" if os.name == "nt" else "clear")


class Console:
    """Static helpers for interactive console UI output (not diagnostics)."""

    @staticmethod
    def info(message: str = "") -> None:
        """Print an interactive UI message.

        Args:
            message: Text to display. Empty string prints a blank line.
        """
        print(message)

    @staticmethod
    def display(value: Any) -> None:
        """Print a value (e.g. DataFrame or array) in the interactive UI.

        Args:
            value: Object to display.
        """
        print(value)

    @staticmethod
    def outline(content: str, logger: Logger | None = None) -> None:
        """Print a timestamped UI message and optionally mirror it to a logger.

        Args:
            content: Message to display.
            logger: Optional logger that receives the same message via ``info``.
        """
        now_utc = datetime.now()
        now = now_utc.astimezone(timezone)
        outline = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {content}"
        print(outline)
        if logger is not None:
            logger.info(outline)

    @staticmethod
    def highlight(message: str) -> None:
        """Clear the screen and print a highlighted section header.

        Args:
            message: Header text to display.
        """
        clear_screen()
        print(f"\n{COLOR_YELLOW}{message}{END_COLOR}")
        print(f"\n{COLOR_YELLOW}{'*' * HIGHLIGHT_SEPARATOR_LEN}{END_COLOR}\n")

    @staticmethod
    def stop_continue(message: str) -> None:
        """Pause execution until the user presses Enter.

        Args:
            message: Prompt shown while waiting.
        """
        input(f"\n{COLOR_GREEN}{message}{END_COLOR}\n")
