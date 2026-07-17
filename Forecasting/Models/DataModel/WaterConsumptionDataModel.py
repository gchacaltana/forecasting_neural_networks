#!/usr/bin/env python3
"""
Data-access layer for monthly water-consumption records.

Queries the database (via stored procedures) for consumption history of a
given building and apartment, used as input for the forecasting pipeline.
"""
from __future__ import annotations

from typing import Any

from Forecasting.Settings.DBConnect import DBConnect


class WaterConsumptionDataModel:
    """Data model for monthly water consumption records."""

    def __init__(self) -> None:
        """Initialize the database connection helper."""
        self.db_connect = DBConnect()

    def get_wm_month_consumption_by_property(
        self, community_code: str, property_name: str
    ) -> list[dict[str, Any]]:
        """Fetch monthly water consumption for a building and apartment.

        Opens a short-lived connection for the query and closes it afterward.

        Args:
            community_code: Building / community identifier.
            property_name: Apartment or property name.

        Returns:
            List of consumption records as dictionaries. Empty if no rows.
        """
        with self.db_connect.db_cursor() as cursor:
            cursor.callproc(
                "dbwaterc.sp_get_monthly_wc_by_property",
                (community_code, property_name),
            )
            for result_set in cursor.stored_results():
                return result_set.fetchall()
        return []
