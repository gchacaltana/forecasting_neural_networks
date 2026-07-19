#!/usr/bin/env python3
"""
Data-access layer for monthly water-consumption records.

Loads consumption history from a JSON dataset for a given building and
apartment, used as input for the forecasting pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.constants import WC_DATA_FILE


class WaterConsumptionDataModel:
    """Data model for monthly water consumption records."""

    def __init__(self, data_file: str | Path | None = None) -> None:
        """Load the JSON dataset used as the consumption source.

        Args:
            data_file: Optional path to the JSON file. Defaults to ``WC_DATA_FILE``.
        """
        path = Path(data_file) if data_file is not None else Path(WC_DATA_FILE)
        if not path.is_absolute():
            path = Path.cwd() / path
        self.data_file = path
        with self.data_file.open(encoding="utf-8") as file:
            self._dataset: dict[str, Any] = json.load(file)

    def get_wm_month_consumption_by_property(
        self, community_code: str, property_name: str
    ) -> list[dict[str, Any]]:
        """Fetch monthly water consumption for a building and apartment.

        Args:
            community_code: Building / community identifier.
            property_name: Apartment or property name.

        Returns:
            List of consumption records as dictionaries. Empty if not found.
        """
        building = self._dataset.get(community_code, {})
        records = building.get(property_name, [])
        return list(records)
