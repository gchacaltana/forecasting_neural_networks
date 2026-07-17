#!/usr/bin/env python3
"""
Shared application constants.

Centralizes UI, plotting, and neural-network defaults so magic numbers are
not scattered across modules.
"""

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
TF_CPP_MIN_LOG_LEVEL = "3"

# Neural network defaults
DEFAULT_EPOCHS = 60
DEFAULT_TRAIN_PREDICTIVE_MONTHS = 3
DEFAULT_TRAIN_PERCENTAGE = 80
DEFAULT_TEST_PERCENTAGE = 20
DEFAULT_ACTIVATION = "tanh"
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_LOSS = "mean_absolute_error"
FEATURE_RANGE = (-1, 1)
PREDICTION_DECIMAL_PLACES = 3
