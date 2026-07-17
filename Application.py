#!/usr/bin/env python3
"""
CLI entry point for the water-consumption forecasting application.

Parses command-line arguments and dispatches to the training/prediction
workflow (e.g. ``python Application.py train``).
"""

import logging
import sys

from core import AppError, MissingArgumentError
from core.logging_config import configure_logging
from Forecasting.Settings.ApplicationConfig import ApplicationConfig
from WCTrainPredict import WCTrainPredict

configure_logging()
logger = logging.getLogger(__name__)


class Application:
    """Orchestrates CLI dispatch for water-consumption forecasting commands."""

    def __init__(self, argv: list[str]) -> None:
        """Initialize the application and dispatch the requested command.

        Args:
            argv: Command-line arguments, typically ``sys.argv``.
        """
        self.argv = argv
        self.app = ApplicationConfig()
        self.dispatcher()

    def dispatcher(self) -> None:
        """Validate CLI arguments and route to the matching workflow."""
        if len(sys.argv) <= 1:
            raise MissingArgumentError("First parameter is not defined")
        self.wm_train()

    def wm_train(self) -> None:
        """Start neural-network training when the ``train`` command is given."""
        if self.argv[1] == "train":
            wctp = WCTrainPredict()
            wctp.run()


# Main entry point
if __name__ == "__main__":
    try:
        wmapp = Application(sys.argv)
    except AppError as err:
        logger.error("Application error: %s", err)
