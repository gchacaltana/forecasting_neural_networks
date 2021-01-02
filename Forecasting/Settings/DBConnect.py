# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"
from Forecasting.Settings.ApplicationConfig import ApplicationConfig
import mysql.connector
import sys

class DBConnect():
    def __init__(self):
        self.app_config = ApplicationConfig()

    def connect_db(self):
        self.mydb = mysql.connector.connect(
            host = self.app_config.db_conection['host'],
            user = self.app_config.db_conection['usr'],
            password = self.app_config.db_conection['pwd'],
            database = self.app_config.db_conection['db']
        )
        return self.mydb