# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entry point for the water-consumption forecasting application.

Parses command-line arguments and dispatches to the training/prediction
workflow (e.g. ``python Application.py train``).
"""
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"

from Forecasting.Settings.ApplicationConfig import ApplicationConfig
from WCTrainPredict import WCTrainPredict
from core import AppError, MissingArgumentError
import sys,os

class Application(object):
    """Orchestrates CLI dispatch for water-consumption forecasting commands."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.app = ApplicationConfig()
        self.dispatcher()

    def dispatcher(self) -> None:
        if len(sys.argv) <= 1:
            raise MissingArgumentError("First parameter is not defined")
        self.wm_train()

    def wm_train(self) -> None:
        """
        Train the neural network for forecasting.
        """
        if self.argv[1] == "train":
            wctp = WCTrainPredict()
            wctp.run()


# Main entry point
if __name__ == "__main__":
    try:
        wmapp = Application(sys.argv)
    except AppError as err:
        print("Exception: ", err)
