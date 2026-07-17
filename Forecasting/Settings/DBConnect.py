#!/usr/bin/env python3
"""
MySQL connection factory for the forecasting application.

Reads database credentials from ``ApplicationConfig`` and returns an open
``mysql.connector`` connection used by data-model classes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mysql.connector

from Forecasting.Settings.ApplicationConfig import ApplicationConfig

if TYPE_CHECKING:
    from mysql.connector.connection import MySQLConnection


class DBConnect:
    """Create MySQL connections from application configuration."""

    def __init__(self) -> None:
        """Load application configuration used for database credentials."""
        self.app_config = ApplicationConfig()

    def connect_db(self) -> MySQLConnection:
        """Open and return a MySQL connection using configured credentials.

        Returns:
            An active ``mysql.connector`` connection.
        """
        self.mydb = mysql.connector.connect(
            host=self.app_config.db_conection['host'],
            user=self.app_config.db_conection['usr'],
            password=self.app_config.db_conection['pwd'],
            database=self.app_config.db_conection['db'],
        )
        return self.mydb
