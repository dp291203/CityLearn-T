from typing import List, Dict
from citylearn.base import Environment

class SecondaryStorage(Environment):
    def __init__(
        self,
        capacity: float,
        max_charge_rate: float,
        max_discharge_rate: float,
        efficiency: float = 0.95,
        initial_soc: float = 0.0,
        **kwargs
    ):
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        self.__soc = [initial_soc]
        self.__charge = [0.0]
        self.__discharge = [0.0]
        self.__energy_distributed = [0.0]
        
        super().__init__(**kwargs)

    @property
    def soc(self) -> List[float]:
        return self.__soc

    @property
    def capacity(self) -> float:
        """Maximum energy storage capacity in kWh."""
        return self.__capacity

    @property
    def max_charge_rate(self) -> float:
        """Maximum charging power in kW."""
        return self.__max_charge_rate

    @property
    def max_discharge_rate(self) -> float:
        """Maximum discharging power in kW."""
        return self.__max_discharge_rate

    @property
    def efficiency(self) -> float:
        """Round-trip efficiency of the storage."""
        return self.__efficiency

    @property
    def soc(self) -> List[float]:
        """State of charge time series in kWh."""
        return self.__soc

    @property
    def charge(self) -> List[float]:
        """Charging power time series in kW."""
        return self.__charge

    @property
    def discharge(self) -> List[float]:
        """Discharging power time series in kW."""
        return self.__discharge

    @property
    def energy_distributed(self) -> List[float]:
        """Energy distributed to buildings time series in kWh."""
        return self.__energy_distributed

    @capacity.setter
    def capacity(self, capacity: float):
        assert capacity > 0, 'capacity must be > 0.'
        self.__capacity = capacity

    @max_charge_rate.setter
    def max_charge_rate(self, max_charge_rate: float):
        assert max_charge_rate >= 0, 'max_charge_rate must be >= 0.'
        self.__max_charge_rate = max_charge_rate

    @max_discharge_rate.setter
    def max_discharge_rate(self, max_discharge_rate: float):
        assert max_discharge_rate >= 0, 'max_discharge_rate must be >= 0.'
        self.__max_discharge_rate = max_discharge_rate

    @efficiency.setter
    def efficiency(self, efficiency: float):
        assert 0 < efficiency <= 1, 'efficiency must be > 0 and <= 1.'
        self.__efficiency = efficiency

    def charge_from_buildings(self, buildings: List['Building']):
        """Charge storage from building surplus energy.
        
        Parameters
        ----------
        buildings : List[Building]
            List of buildings in the district
        """
        total_surplus = sum(b.shared_energy[self.time_step] for b in buildings)
        
        # Calculate how much we can actually charge based on capacity and rate limits
        max_charge_energy = min(
            self.max_charge_rate * self.seconds_per_time_step / 3600,
            self.capacity - self.soc[self.time_step]
        )
        
        actual_charge = min(total_surplus, max_charge_energy)
        
        # Update storage state
        self.__charge[self.time_step] = actual_charge * 3600 / self.seconds_per_time_step  # Convert to kW
        self.__soc[self.time_step] += actual_charge * self.efficiency
        
        # Distribute the charging across buildings proportionally
        if total_surplus > 0:
            for building in buildings:
                share = building.shared_energy[self.time_step] / total_surplus
                building.shared_energy[self.time_step] -= share * actual_charge

    def discharge_to_buildings(self, buildings: List['Building']):
        """Discharge storage to deficit buildings.
        
        Parameters
        ----------
        buildings : List[Building]
            List of buildings in the district
        """
        deficit_buildings = [b for b in buildings if b.net_electricity_consumption[self.time_step] > 0]
        total_deficit = sum(b.net_electricity_consumption[self.time_step] for b in deficit_buildings)
        
        if not deficit_buildings or total_deficit <= 0:
            self.__discharge[self.time_step] = 0.0
            self.__energy_distributed[self.time_step] = 0.0
            return
        
        # Calculate how much we can actually discharge
        max_discharge_energy = min(
            self.max_discharge_rate * self.seconds_per_time_step / 3600,
            self.soc[self.time_step]
        )
        
        # Distribute proportionally to deficit buildings
        distributed_energy = 0.0
        for building in deficit_buildings:
            share = building.net_electricity_consumption[self.time_step] / total_deficit
            energy_to_building = min(max_discharge_energy * share, 
                                   building.net_electricity_consumption[self.time_step])
            
            building.net_electricity_consumption[self.time_step] -= energy_to_building
            distributed_energy += energy_to_building
            
            if distributed_energy >= max_discharge_energy:
                break
        
        # Update storage state
        self.__discharge[self.time_step] = distributed_energy * 3600 / self.seconds_per_time_step  # Convert to kW
        self.__soc[self.time_step] -= distributed_energy / self.efficiency
        self.__energy_distributed[self.time_step] = distributed_energy

    def next_time_step(self):
        """Advance to next time step."""
        self.__soc.append(self.__soc[self.time_step])
        self.__charge.append(0.0)
        self.__discharge.append(0.0)
        self.__energy_distributed.append(0.0)
        super().next_time_step()

    def reset(self):
        """Reset storage to initial state."""
        self.__soc = [self.__soc[0]]
        self.__charge = [0.0]
        self.__discharge = [0.0]
        self.__energy_distributed = [0.0]
        super().reset()

    def __str__(self):
        return (
            f"Secondary Storage:\n"
            f"  Capacity: {self.capacity} kWh\n"
            f"  Max Charge Rate: {self.max_charge_rate} kW\n"
            f"  Max Discharge Rate: {self.max_discharge_rate} kW\n"
            f"  Efficiency: {self.efficiency * 100}%\n"
            f"  Current SOC: {self.soc[self.time_step]:.2f} kWh ({self.soc[self.time_step]/self.capacity*100:.1f}%)"
        )