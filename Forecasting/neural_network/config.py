#!/usr/bin/env python3
"""
Configuration for the feed-forward water-consumption forecasting network.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.constants import (
    DEFAULT_ACTIVATION,
    DEFAULT_EPOCHS,
    DEFAULT_LOSS,
    DEFAULT_OPTIMIZER,
    DEFAULT_TEST_PERCENTAGE,
    DEFAULT_TRAIN_PERCENTAGE,
    DEFAULT_TRAIN_PREDICTIVE_MONTHS,
)


@dataclass
class NeuralNetworkConfig:
    """Hyperparameters for training and building the MLP forecaster."""

    epochs: int = DEFAULT_EPOCHS
    train_predictive_months: int = DEFAULT_TRAIN_PREDICTIVE_MONTHS
    train_percentage: int = DEFAULT_TRAIN_PERCENTAGE
    test_percentage: int = DEFAULT_TEST_PERCENTAGE
    activation: str = DEFAULT_ACTIVATION
    optimizer: str = DEFAULT_OPTIMIZER
    loss: str = DEFAULT_LOSS
    hidden_neurons: int | None = None
