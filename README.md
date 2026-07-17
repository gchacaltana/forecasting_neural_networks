# Forecasting with neural networks

## Overview

Demo application that implements the training process of a neural network for forecasting monthly drinking-water consumption of a household.

## Dataset

The dataset corresponds to drinking-water consumption (in m3) of a household. The variables are: Billing Date, Year, Month, Consumption M3, and Total Amount.

Sample data structure:

    2017-11-11         2017   11  27.957   S/. 42.63
    2017-12-12         2017   12  27.961   S/. 32.09
    2018-01-11         2018    1  31.906   S/. 41.17
    2018-02-10         2018    2  38.824   S/. 47.41
    2018-03-13         2018    3  38.417   S/. 60.94
    2018-04-13         2018    4  33.981   S/. 67.16
    2018-05-12         2018    5  39.217   S/. 78.19
    2018-06-12         2018    6  40.855   S/. 76.69
    2018-07-13         2018    7  38.783   S/. 72.53
    2018-08-11         2018    8  32.925   S/. 68.48
    2018-09-11         2018    9  37.412   S/. 80.35
    2018-10-11         2018   10  42.413   S/. 77.30
    2018-11-10         2018   11  45.599   S/. 85.96
    2018-12-12         2018   12  47.315  S/. 122.53
    2019-01-10         2019    1  28.595   S/. 75.69
    2019-02-09         2019    2  38.238  S/. 105.69
    2019-03-12         2019    3  46.413  S/. 130.57
    2019-04-11         2019    4  37.242  S/. 104.81
    2019-05-11         2019    5  29.696   S/. 83.58
    2019-06-12         2019    6  36.779  S/. 103.57
    2019-07-11         2019    7  32.506   S/. 79.34
    2019-08-12         2019    8  32.848   S/. 88.09
    2019-09-10         2019    9  35.495  S/. 100.91
    2019-10-11         2019   10  38.442  S/. 109.31
    2019-11-11         2019   11  40.088  S/. 115.42
    2019-12-11         2019   12  35.821  S/. 103.13
    2020-01-10         2020    1  35.784  S/. 103.02
    2020-02-10         2020    2  35.368  S/. 101.80
    2020-03-11         2020    3  41.188  S/. 118.52
    2020-04-11         2020    4  34.664   S/. 92.39
    2020-05-12         2020    5  32.021   S/. 90.84
    2020-06-10         2020    6  30.789   S/. 88.68
    2020-07-10         2020    7  29.120   S/. 87.96
    2020-08-10         2020    8  29.894   S/. 91.42
    2020-09-10         2020    9  27.399   S/. 78.88
    2020-10-10         2020   10  25.238   S/. 72.69
    2020-11-11         2020   11  22.726   S/. 65.43

## Neural network configuration

    Build a feed-forward neural network model.
    Architecture:
        * 1 hidden layer with "n" neurons (entered via console)
        * 1 output neuron
    Activation function: Hyperbolic Tangent (for normalized values in [-1, 1])
    Optimizer: Adam
    Loss metric: Mean Absolute Error (MAE)
    Accuracy metric: Mean Squared Error (MSE)

## Requirements

Dependencies are declared in `pyproject.toml`. Install them with:

```bash
pip install -e .
```

For development tools (ruff, mypy):

```bash
pip install -e ".[dev]"
```

Pinned runtime versions:

```bash
numpy 1.19.3
pandas 1.0.1
Keras 2.4.3
tensorflow 2.4.0
scikit-learn 0.22.1
matplotlib 3.1.3
python-dotenv
mysql-connector-python
pytz
```

## How to run

```bash
python Application.py train
```
