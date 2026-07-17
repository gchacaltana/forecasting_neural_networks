# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"
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
