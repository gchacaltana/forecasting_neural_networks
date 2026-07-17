#!/usr/bin/env python3
"""
Shared application core package.

Re-exports exceptions. Constants: ``core.constants``.
Logging setup: ``core.logging_config``.
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
