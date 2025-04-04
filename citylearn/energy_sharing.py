import inspect
import math
import time
import logging
import random
import importlib
import os
from pathlib import Path
from enum import Enum, unique
from typing import Any, List, Mapping, Tuple, Union, Optional, Dict

import numpy as np
import pandas as pd
import folium

from gym import Env, spaces
from gym.core import RenderFrame

from citylearn import __version__ as citylearn_version
from citylearn.base import Environment
from citylearn.building import Building
from citylearn.electric_vehicle import electric_vehicle
from citylearn.charger import Charger
from citylearn.cost_function import CostFunction
from citylearn.data import DataSet, EnergySimulation, CarbonIntensity, Pricing, Weather, EVSimulation
from citylearn.utilities import read_json

class Energy_Sharing:
    f"""Energy sharing class."""

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        f"""Calculate the great circle distance between two points on the earth (specified in decimal degrees)."""
        R = 6371  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    
    @staticmethod
    def surplus(self) -> float:
        """Returns the total surplus energy available for sharing."""
        net_consumption = np.array(self.net_electricity_consumption)
        return np.sum(np.where(net_consumption < 0, -net_consumption, 0))  # Sum of surplus energy

    @staticmethod
    def get_reservable_energy(self) -> float:
        """Computes the energy that a surplus building should reserve before sharing excess energy."""
        battery_reserve = 0.0
        battery = getattr(self, 'electrical_storage', None)

        if battery:
            min_soc_p = 0.2  # Maintain at least 20% SOC
            current_soc = battery.soc[self.time_step]
            battery_capacity = battery.capacity
            current_soc_p = current_soc / battery_capacity

            if min_soc_p > current_soc_p:
                battery_reserve = 0.0
            else:
                battery_reserve = min_soc_p * battery_capacity

        total_ev_demand = 0.0
        chargers = getattr(self, 'chargers', [])

        if chargers:
            for charger in chargers:
                if charger is not None and charger.connected_ev is not None:
                    ev = charger.connected_ev
                    required_energy = ev.battery.capacity * (
                        ev.ev_simulation.required_soc_departure[self.time_step] / 100 - ev.battery.soc[self.time_step] / 100
                    )
                    max_charge_power = min(charger.max_charging_power, required_energy)
                    total_ev_demand += max(0, max_charge_power)

        return battery_reserve + total_ev_demand

    @staticmethod
    def max_shared_energy(self) -> float:
        """Computes the maximum energy that can be shared with other buildings."""
        surplus = self.surplus()
        reserved_energy = self.get_reservable_energy()
        max_shareable_energy = max(0, surplus - reserved_energy)
        return max_shareable_energy
    
    def energy_sharing
