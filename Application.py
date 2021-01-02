# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Application Main
"""
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"

from Forecasting.Settings.ApplicationConfig import ApplicationConfig
from WCTrainPredict import WCTrainPredict
import sys,os

class Application(object):
    """
    Main Application Water Consumption Predict
    """

    def __init__(self, argv):
        self.argv = argv
        self.app = ApplicationConfig()
        self.dispatcher()

    def dispatcher(self):
        if len(sys.argv) <= 1:
            raise Exception("Primer parametro no definido")
        self.wm_train()

    def wm_train(self):
        """
        Entrenar red neuronal para pronóstico
        """
        if self.argv[1] == "train":
            wctp = WCTrainPredict()
            wctp.run()


# Función Main
if __name__ == "__main__":
    try:
        wmapp = Application(sys.argv)
    except (Exception, TypeError, IndexError) as err:
        print("Exception: ", err)
