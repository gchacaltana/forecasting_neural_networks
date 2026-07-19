#!/usr/bin/env python3
"""
Shared application core package.
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
