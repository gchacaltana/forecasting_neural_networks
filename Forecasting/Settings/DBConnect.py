#!/usr/bin/env python3
"""
MySQL connection factory for the forecasting application.

Reads database credentials from ``ApplicationConfig`` and returns an open
``mysql.connector`` connection used by data-model classes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from Forecasting.Settings.ApplicationConfig import ApplicationConfig
import mysql.connector
import sys

if TYPE_CHECKING:
    from mysql.connector.connection import MySQLConnection

class DBConnect():
    def __init__(self) -> None:
        self.app_config = ApplicationConfig()

    def connect_db(self) -> MySQLConnection:
        self.mydb = mysql.connector.connect(
            host = self.app_config.db_conection['host'],
            user = self.app_config.db_conection['usr'],
            password = self.app_config.db_conection['pwd'],
            database = self.app_config.db_conection['db']
        )
        return self.mydb
