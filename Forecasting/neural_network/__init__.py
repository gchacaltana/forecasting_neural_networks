#!/usr/bin/env python3
"""
Neural network package for water-consumption forecasting.

Contains hyperparameters configuration and the interactive train/predict
pipeline (feed-forward MLP over a time-series window).
"""
from __future__ import annotations

from forecasting.neural_network.config import NeuralNetworkConfig
from forecasting.neural_network.train_predict import WCTrainPredict

__all__ = [
    "NeuralNetworkConfig",
    "WCTrainPredict",
]
