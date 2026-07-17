#!/usr/bin/env python3
"""
Shared application core package.

Re-exports cross-cutting building blocks (currently the custom exception
hierarchy) so other modules can import them from ``core`` directly.
"""
from __future__ import annotations

from core.exceptions import (
    AppError,
    DatasetNotFoundError,
    InvalidPartitionError,
    MissingArgumentError,
)

__all__ = [
    "AppError",
    "DatasetNotFoundError",
    "InvalidPartitionError",
    "MissingArgumentError",
]
