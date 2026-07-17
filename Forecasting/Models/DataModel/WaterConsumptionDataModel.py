# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"

from typing import Any

from Forecasting.Settings.DBConnect import DBConnect

class WaterConsumptionDataModel(object):
    """
    Clase para obtener información sobre los consumos mensuales de agua.
    """
    
    def __init__(self) -> None:
        db_connect = DBConnect()
        self.dbwm_connection = db_connect.connect_db()
        self.dbwm_cursor = self.dbwm_connection.cursor(dictionary = True)

    def get_wm_month_consumption_by_property(
        self, community_code: str, property_name: str
    ) -> list[dict[str, Any]] | None:
        self.dbwm_cursor.callproc("dbwaterc.sp_get_monthly_wc_by_property",(community_code, property_name))
        results = self.dbwm_cursor.stored_results()
        for results in self.dbwm_cursor.stored_results():
            return results.fetchall()
