import inspect
import math
from typing import Any, List, Mapping, Tuple, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.energy_model import Battery


class SecondaryStorage(Environment):
    """Centralized battery storage for energy sharing between buildings.

    Works like the electric_vehicle pattern: the Battery object owns the
    soc list, and ``battery.charge()`` is called every time step so that
    ``battery.soc[time_step]`` is always valid.

    Sign convention (same as the rest of CityLearn):
        * **net_electricity_consumption < 0** → building has *surplus*
        * **net_electricity_consumption > 0** → building has *deficit*
        * **action > 0** → charge the centralized battery
        * **action < 0** → discharge the centralized battery

    Parameters
    ----------
    battery : Battery
        Battery object for energy storage.
    observation_metadata : dict
        Mapping of active and inactive observations.
    action_metadata : dict
        Mapping of active and inactive actions.
    name : str, optional
        Unique secondary storage name.
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(
        self,
        battery: Battery,
        observation_metadata: Mapping[str, bool],
        action_metadata: Mapping[str, bool],
        name: str = None,
        **kwargs: Any
    ):
        self.name = name or "SecondaryStorage"
        self.battery = battery
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.__observation_epsilon = 0.0

        # observation / action spaces
        self.non_periodic_normalized_observation_space_limits = None
        self.periodic_normalized_observation_space_limits = None
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()

        # per-timestep tracking (mirrors Battery.energy_balance / soc)
        self.__electricity_consumption = [0.0]
        self.__energy_balance = [0.0]
        self.__net_energy_flow = [0.0]

        # initialize parent class
        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key: value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique secondary storage name."""
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def battery(self) -> Battery:
        """Battery object for energy storage."""
        return self.__battery

    @battery.setter
    def battery(self, battery: Battery):
        self.__battery = Battery(0.0, 0.0) if battery is None else battery

    @property
    def observation_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive observations."""
        return self.__observation_metadata

    @observation_metadata.setter
    def observation_metadata(self, observation_metadata: Mapping[str, bool]):
        self.__observation_metadata = observation_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive actions."""
        return self.__action_metadata

    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata

    @property
    def observation_space(self) -> spaces.Box:
        """Agent observation space."""
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space

    @property
    def action_space(self) -> spaces.Box:
        """Agent action space."""
        return self.__action_space

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @property
    def active_observations(self) -> List[str]:
        """Observations in observation_metadata with True value."""
        return [k for k, v in self.observation_metadata.items() if v]

    @property
    def active_actions(self) -> List[str]:
        """Actions in action_metadata with True value."""
        return [k for k, v in self.action_metadata.items() if v]

    @property
    def electricity_consumption(self) -> List[float]:
        """Energy consumption time series, in [kWh]."""
        return self.__electricity_consumption

    @electricity_consumption.setter
    def electricity_consumption(self, electricity_consumption: List[float]):
        self.__electricity_consumption = electricity_consumption

    @property
    def energy_balance(self) -> List[float]:
        """Energy balance time series, in [kWh]."""
        return self.__energy_balance

    @property
    def net_energy_flow(self) -> List[float]:
        """Net energy flow time series (positive = charging, negative = discharging), in [kWh]."""
        return self.__net_energy_flow

    @property
    def soc(self) -> float:
        """Current state of charge as a fraction (0-1).

        Uses ``battery.soc[time_step]`` just like the EV pattern so the
        value is always in sync with the simulation clock.
        """
        try:
            soc_kwh = self.battery.soc[self.time_step]
        except (IndexError, TypeError):
            soc_kwh = self.battery.soc[-1] if self.battery.soc else 0.0
        return soc_kwh / self.battery.capacity if self.battery.capacity > 0 else 0.0

    @property
    def soc_kwh(self) -> float:
        """Current state of charge in kWh."""
        try:
            return self.battery.soc[self.time_step]
        except (IndexError, TypeError):
            return self.battery.soc[-1] if self.battery.soc else 0.0

    @property
    def capacity(self) -> float:
        """Battery capacity in kWh."""
        return self.battery.capacity

    @property
    def nominal_power(self) -> float:
        """Battery nominal power in kW."""
        return self.battery.nominal_power

    @property
    def soc_history(self) -> List[float]:
        """Full soc list from the battery (in kWh), one entry per time step."""
        return self.battery.soc

    # ------------------------------------------------------------------
    # actions  (mirrors how charger calls ev.battery.charge directly)
    # ------------------------------------------------------------------

    def apply_actions(self, secondary_storage_action: float = 0.0, **kwargs):
        """Charge / discharge the battery for the current time step.

        Parameters
        ----------
        secondary_storage_action : float, default: 0.0
            Fraction of battery capacity to charge (+) or discharge (-).
            Clamped to [-1, 1].
        """
        if 'secondary_storage_action' in kwargs:
            secondary_storage_action = kwargs['secondary_storage_action']

        secondary_storage_action = max(-1.0, min(1.0, secondary_storage_action))

        # convert fraction → kWh
        energy = secondary_storage_action * self.battery.capacity

        # Battery.charge handles all limits (capacity, power, efficiency,
        # degradation) and appends to battery.soc & battery.energy_balance.
        self.battery.charge(energy)

        # record the *actual* energy that went in / out
        actual = self.battery.energy_balance[-1]
        self.__electricity_consumption[self.time_step] = abs(actual)
        self.__energy_balance[self.time_step] = actual
        self.__net_energy_flow[self.time_step] = actual

    # ------------------------------------------------------------------
    # time step management  (mirrors electric_vehicle.next_time_step)
    # ------------------------------------------------------------------

    def next_time_step(self):
        """Advance to the next time step.

        Calls ``battery.next_time_step()`` first (just like the EV does)
        so that the battery's internal electricity_consumption list stays
        in sync, then advances our own clock and appends zeros to our
        tracking lists for the new step.
        """
        self.battery.next_time_step()
        super().next_time_step()
        self.__electricity_consumption.append(0.0)
        self.__energy_balance.append(0.0)
        self.__net_energy_flow.append(0.0)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def get_observations(self, include_all: bool = None, normalize: bool = None,
                         periodic_normalization: bool = None) -> Mapping[str, Union[int, float]]:
        """Get secondary storage observations.

        Parameters
        ----------
        include_all : bool, default: False
            Whether to include all observations.
        normalize : bool, default: False
            Whether to normalize observations.
        periodic_normalization : bool, default: False
            Whether to apply periodic normalization.

        Returns
        -------
        dict
            Observation name to value mapping.
        """
        observations = {
            'secondary_storage_soc': self.soc,
            'secondary_storage_capacity': self.battery.capacity,
            'secondary_storage_nominal_power': self.battery.nominal_power,
            'secondary_storage_energy_balance': self.__energy_balance[self.time_step],
        }

        if include_all:
            observations.update({
                'secondary_storage_electricity_consumption': self.__electricity_consumption[self.time_step],
                'secondary_storage_net_energy_flow': self.__net_energy_flow[self.time_step],
            })

        return observations

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------

    def estimate_observation_space(self) -> spaces.Box:
        """Get estimate of observation space."""
        low_limit, high_limit = [], []

        for key in self.active_observations:
            if key == 'secondary_storage_soc':
                low_limit.append(0.0)
                high_limit.append(1.0)
            elif key == 'secondary_storage_capacity':
                low_limit.append(0.0)
                high_limit.append(self.battery.capacity * 1.1)
            elif key == 'secondary_storage_nominal_power':
                low_limit.append(0.0)
                high_limit.append(self.battery.nominal_power * 1.1)
            elif key in ['secondary_storage_energy_balance',
                         'secondary_storage_electricity_consumption',
                         'secondary_storage_net_energy_flow']:
                low_limit.append(-self.battery.nominal_power)
                high_limit.append(self.battery.nominal_power)
            else:
                low_limit.append(-1.0)
                high_limit.append(1.0)

        return spaces.Box(
            low=np.array(low_limit, dtype='float32'),
            high=np.array(high_limit, dtype='float32'),
        )

    def estimate_action_space(self) -> spaces.Box:
        """Get estimate of action space."""
        low_limit, high_limit = [], []

        for key in self.active_actions:
            if key == 'secondary_storage':
                low_limit.append(-1.0)
                high_limit.append(1.0)
            else:
                low_limit.append(-1.0)
                high_limit.append(1.0)

        return spaces.Box(
            low=np.array(low_limit, dtype='float32'),
            high=np.array(high_limit, dtype='float32'),
        )

    # ------------------------------------------------------------------
    # autosize / reset
    # ------------------------------------------------------------------

    def autosize_battery(self, **kwargs):
        """Autosize battery for secondary storage."""
        if hasattr(self.battery, 'autosize'):
            self.battery.autosize(**kwargs)

    def reset(self):
        """Reset secondary storage to initial state."""
        super().reset()
        self.battery.reset()
        self.__electricity_consumption = [0.0]
        self.__energy_balance = [0.0]
        self.__net_energy_flow = [0.0]

    # ------------------------------------------------------------------
    # default metadata
    # ------------------------------------------------------------------

    @staticmethod
    def get_default_observation_metadata() -> Mapping[str, bool]:
        """Get default observation metadata for secondary storage."""
        return {
            'secondary_storage_soc': True,
            'secondary_storage_capacity': False,
            'secondary_storage_nominal_power': False,
            'secondary_storage_energy_balance': True,
            'secondary_storage_electricity_consumption': False,
            'secondary_storage_net_energy_flow': False,
        }

    @staticmethod
    def get_default_action_metadata() -> Mapping[str, bool]:
        """Get default action metadata for secondary storage."""
        return {
            'secondary_storage': True,
        }
