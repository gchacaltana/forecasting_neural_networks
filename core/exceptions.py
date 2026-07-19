#!/usr/bin/env python3
"""
Custom exception hierarchy for domain and validation errors.
"""
from __future__ import annotations


class AppError(Exception):
    """Base exception for all application errors."""


class MissingArgumentError(AppError):
    """Raised when a required command-line argument is missing."""


class DatasetNotFoundError(AppError):
    """Raised when no water consumption records exist for the given property."""


class InvalidPartitionError(AppError):
    """Raised when train/test percentage split does not add up to 100."""
