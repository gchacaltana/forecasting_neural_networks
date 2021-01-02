# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"
import socket, os
from dotenv import load_dotenv

class ApplicationConfig(object):
    def __init__(self):
        load_dotenv()
        self.load_config()

    def load_config(self):
        self.hostname = socket.getfqdn()
        self.path_app = os.getcwd()
        self.load_config_prod()
        self.load_config_dev()

    def load_config_prod(self):
        if (self.hostname == os.environ.get("hs_pro")):
            pass

    def load_config_dev(self):
        if (self.hostname == os.environ.get("hs_dev")):
            self.api_token = os.environ.get("api-token")
            self.db_conection = {
                "host": os.environ.get("db.host"),
                "usr": os.environ.get("db.user"),
                "pwd": os.environ.get("db.pswd"),
                "db": os.environ.get("db.name")
            }
