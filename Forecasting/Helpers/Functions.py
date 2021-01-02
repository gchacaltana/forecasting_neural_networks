# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"
from datetime import datetime
import pytz
import pandas as pd
import os
timezone = pytz.timezone("America/Lima")

# Console Constants
HYPHEN_LEN = 70
COLOR_YELLOW = "\033[93m"
END_COLOR = "\033[0m"
COLOR_GREEN = "\033[92m"

class Console():    

    @staticmethod
    def outline(content, logging = None):
        now_utc = datetime.now()
        now = now_utc.astimezone(timezone)
        outline = "{} - {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), content)
        print(outline)
        if logging:
            logging.info(outline)

    @staticmethod
    def highlight(message: str):
        os.system("clear")
        print("\n{}".format(COLOR_YELLOW + message + END_COLOR))
        print("\n{}\n".format(COLOR_YELLOW + "*"*50 + END_COLOR))

    @staticmethod
    def stop_continue(message: str):
        input("\n{}\n".format(COLOR_GREEN + message + END_COLOR))
