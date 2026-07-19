#!/usr/bin/env python3
"""
Shared application constants.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Console / UI
COLOR_YELLOW = "\033[93m"
END_COLOR = "\033[0m"
COLOR_GREEN = "\033[92m"
HIGHLIGHT_SEPARATOR_LEN = 50
TIMEZONE_NAME = "America/Lima"

# Plotting
PLT_FIGURE_SIZE = (16, 9)
PLT_STYLE = "fast"

# TensorFlow logging (must be applied before importing keras/tensorflow)
TF_CPP_MIN_LOG_LEVEL = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")

# Neural network defaults (from .env, with fallbacks)
DEFAULT_EPOCHS = int(os.getenv("NN_DEFAULT_EPOCHS", "60"))
DEFAULT_TRAIN_PREDICTIVE_MONTHS = int(
    os.getenv("NN_DEFAULT_TRAIN_PREDICTIVE_MONTHS", "3")
)
DEFAULT_TRAIN_PERCENTAGE = int(os.getenv("NN_DEFAULT_TRAIN_PERCENTAGE", "80"))
DEFAULT_TEST_PERCENTAGE = int(os.getenv("NN_DEFAULT_TEST_PERCENTAGE", "20"))
DEFAULT_ACTIVATION = os.getenv("NN_DEFAULT_ACTIVATION", "tanh")
DEFAULT_OPTIMIZER = os.getenv("NN_DEFAULT_OPTIMIZER", "Adam")
DEFAULT_LOSS = os.getenv("NN_DEFAULT_LOSS", "mean_absolute_error")
FEATURE_RANGE = (
    float(os.getenv("NN_FEATURE_RANGE_MIN", "-1")),
    float(os.getenv("NN_FEATURE_RANGE_MAX", "1")),
)
PREDICTION_DECIMAL_PLACES = int(os.getenv("NN_PREDICTION_DECIMAL_PLACES", "3"))

# Dataset (JSON)
WC_DATA_FILE = os.getenv("WC_DATA_FILE", "data/water_consumption.json")
