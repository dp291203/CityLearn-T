import inspect
import math
from typing import Any, List, Mapping, Tuple, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.energy_model import Battery


class SecondaryStorage(Environment):
    """Dynamic secondary storage system for energy sharing between buildings.
    
    This class implements a centralized battery storage system that can be controlled
    by agents through charge/discharge actions, similar to how EVs are handled.
    
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
        
        # Initialize observation and action spaces
        self.non_periodic_normalized_observation_space_limits = None
        self.periodic_normalized_observation_space_limits = None
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()
        
        # Energy tracking
        self.__electricity_consumption = []
        self.__energy_balance = []
        self.__net_energy_flow = []
        self.__soc_history = []
        
        # Initialize parent class
        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key: value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)

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
        """Energy consumption/generation time series, in [kWh]."""
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
        """Current state of charge as a fraction."""
        battery_soc = self.battery.soc
        if isinstance(battery_soc, list):
            return battery_soc[-1] if battery_soc else 0.0
        return battery_soc

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
        """State of charge history time series."""
        return self.__soc_history

    def charge(self, energy: float) -> float:
        """Charge the secondary storage battery.
        
        Parameters
        ----------
        energy : float
            Energy to charge in kWh (positive value).
            
        Returns
        -------
        float
            Actual energy charged.
        """
        if energy <= 0:
            return 0.0
            
        # Limit charging by available capacity and power constraints
        current_soc = self.soc  # Use the property that handles list/float
        max_charge = min(
            energy,
            self.battery.capacity * (1.0 - current_soc),  # Available capacity
            self.battery.nominal_power  # Power constraint
        )
        
        if max_charge > 0:
            self.battery.charge(max_charge)
            
        return max_charge

    def discharge(self, energy: float) -> float:
        """Discharge the secondary storage battery.
        
        Parameters
        ----------
        energy : float
            Energy to discharge in kWh (positive value).
            
        Returns
        -------
        float
            Actual energy discharged.
        """
        if energy <= 0:
            return 0.0
            
        # Limit discharging by available energy and power constraints
        current_soc = self.soc  # Use the property that handles list/float
        max_discharge = min(
            energy,
            self.battery.capacity * current_soc,  # Available energy
            self.battery.nominal_power  # Power constraint
        )
        
        if max_discharge > 0:
            self.battery.charge(-max_discharge)  # Use negative energy for discharge
            
        return max_discharge

    def apply_actions(self, secondary_storage_action: float = 0.0, **kwargs):
        """Apply secondary storage action.
        
        Parameters
        ----------
        secondary_storage_action : float, default: 0.0
            Fraction of battery capacity to charge/discharge by.
            Positive values = charge, negative values = discharge.
        """
        if 'secondary_storage_action' in kwargs:
            secondary_storage_action = kwargs['secondary_storage_action']
            
        # Convert action to energy amount
        energy_amount = abs(secondary_storage_action) * self.battery.capacity
        
        if secondary_storage_action > 0:
            # Charge
            actual_energy = self.charge(energy_amount)
            net_flow = actual_energy
        elif secondary_storage_action < 0:
            # Discharge
            actual_energy = self.discharge(energy_amount)
            net_flow = -actual_energy
        else:
            net_flow = 0.0
            
        # Update tracking variables
        self.__net_energy_flow.append(net_flow)
        self.__energy_balance.append(net_flow)
        self.__electricity_consumption.append(net_flow)
        self.__soc_history.append(self.soc)

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
            'secondary_storage_soc': self.battery.soc,
            'secondary_storage_capacity': self.battery.capacity,
            'secondary_storage_nominal_power': self.battery.nominal_power,
            'secondary_storage_energy_balance': self.__energy_balance[-1] if self.__energy_balance else 0.0,
        }
        
        if include_all:
            observations.update({
                'secondary_storage_electricity_consumption': self.__electricity_consumption[-1] if self.__electricity_consumption else 0.0,
                'secondary_storage_net_energy_flow': self.__net_energy_flow[-1] if self.__net_energy_flow else 0.0,
            })
            
        return observations

    def estimate_observation_space(self) -> spaces.Box:
        """Get estimate of observation space.
        
        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """
        low_limit, high_limit = [], []
        
        for key in self.active_observations:
            if key == 'secondary_storage_soc':
                low_limit.append(0.0)
                high_limit.append(1.0)
            elif key == 'secondary_storage_capacity':
                low_limit.append(0.0)
                high_limit.append(self.battery.capacity * 1.1)  # 10% buffer
            elif key == 'secondary_storage_nominal_power':
                low_limit.append(0.0)
                high_limit.append(self.battery.nominal_power * 1.1)  # 10% buffer
            elif key in ['secondary_storage_energy_balance', 'secondary_storage_electricity_consumption', 'secondary_storage_net_energy_flow']:
                low_limit.append(-self.battery.nominal_power)
                high_limit.append(self.battery.nominal_power)
            else:
                low_limit.append(-1.0)
                high_limit.append(1.0)
                
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def estimate_action_space(self) -> spaces.Box:
        """Get estimate of action space.
        
        Returns
        -------
        action_space : spaces.Box
            Action low and high limits.
        """
        low_limit, high_limit = [], []
        
        for key in self.active_actions:
            if key == 'secondary_storage':
                # Action is fraction of capacity, bounded by power constraints
                limit = self.battery.nominal_power / self.battery.capacity if self.battery.capacity > 0 else 1.0
                low_limit.append(-limit)
                high_limit.append(limit)
            else:
                low_limit.append(-1.0)
                high_limit.append(1.0)
                
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def autosize_battery(self, **kwargs):
        """Autosize battery for secondary storage.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to battery autosize function.
        """
        # Default sizing for secondary storage
        if hasattr(self.battery, 'autosize'):
            self.battery.autosize(**kwargs)

    def reset(self):
        """Reset secondary storage to initial state."""
        super().reset()
        self.__electricity_consumption = []
        self.__energy_balance = []
        self.__net_energy_flow = []
        self.__soc_history = []
        self.battery.reset()

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
