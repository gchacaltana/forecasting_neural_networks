# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom exception hierarchy for domain and validation errors.

All application-specific failures inherit from ``AppError`` so the CLI entry
point can catch them uniformly and present a clear message to the user.
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
