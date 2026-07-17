#!/usr/bin/env python3
"""
Console helpers for the interactive forecasting CLI.

Provides timestamped output, highlighted section headers, and pause prompts
used throughout the training and prediction workflow.
"""
from __future__ import annotations

from datetime import datetime
from logging import Logger
import pytz
import pandas as pd
import os
timezone = pytz.timezone("America/Lima")

# Console Constants
HYPHEN_LEN = 70
COLOR_YELLOW = "\033[93m"
END_COLOR = "\033[0m"
COLOR_GREEN = "\033[92m"

class Console():    

    @staticmethod
    def outline(content: str, logging: Logger | None = None) -> None:
        now_utc = datetime.now()
        now = now_utc.astimezone(timezone)
        outline = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {content}"
        print(outline)
        if logging:
            logging.info(outline)

    @staticmethod
    def highlight(message: str) -> None:
        os.system("clear")
        print(f"\n{COLOR_YELLOW}{message}{END_COLOR}")
        print(f"\n{COLOR_YELLOW}{'*' * 50}{END_COLOR}\n")

    @staticmethod
    def stop_continue(message: str) -> None:
        input(f"\n{COLOR_GREEN}{message}{END_COLOR}\n")
