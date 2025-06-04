from typing import Mapping, List
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.energy_model import Battery
import copy

class SecondaryStorage(Environment):
    def __init__(self, 
                 observation_metadata: Mapping[str, bool],
                 action_metadata: Mapping[str, bool],
                 battery: Battery = None,
                 name: str = None,
                 **kwargs):
        """
        Central shared battery to store surplus and meet deficits.
        """
        self.name = name
        self.battery = battery
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.__observation_epsilon = 0.0

        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()

        super().__init__(**kwargs)

    # @property
    # def observation_space(self) -> spaces.Box:
    #     """Agent observation space."""

    #     return self.__observation_space

    @property
    def action_space(self) -> spaces.Box:
        """Agent action spaces."""

        return self.__action_space

    def estimate_observation_space(self) -> spaces.Box:
        low = []
        high = []

        if self.observation_metadata.get('soc', False):
            low.append(0.0)
            high.append(1.0)

        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def estimate_action_space(self) -> spaces.Box:
        low = []
        high = []

        if self.action_metadata.get('storage_action', False):
            low.append(-1.0)  # discharge
            high.append(1.0)  # charge

        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    # @observation_space.setter
    # def observation_space(self, observation_space: spaces.Box):
    #     self.__observation_space = observation_space
    #     self.non_periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
    #         include_all=True, periodic_normalization=False
    #     )
    #     self.periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
    #         include_all=True, periodic_normalization=True
    #     )

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @property
    def active_observations(self) -> List[str]:
        return [k for k, v in self.observation_metadata.items() if v]

    @property
    def active_actions(self) -> List[str]:
        return [k for k, v in self.action_metadata.items() if v]

    def observations(self) -> Mapping[str, float]:
        obs = {}

        if self.observation_metadata.get('soc', False):
            obs['soc'] = self.battery.soc[self.time_step] / self.battery.capacity

        return obs

    def apply_action(self, action: float):
        """
        Apply charge or discharge based on agent action.
        """
        max_power = self.battery.nominal_power
        soc = self.battery.soc[self.time_step]
        capacity = self.battery.capacity

        if action > 0 and soc < capacity:
            energy = min(action * max_power, self.battery.capacity - self.battery.soc[self.time_step])
            self.battery.charge(energy)
        elif action < 0:
            energy = min((action) * max_power, self.battery.soc[self.time_step])
            self.battery.charge(energy)

    def next_time_step(self):
        self.battery.next_time_step()
        super().next_time_step()

    def reset(self):
        self.battery.reset()
        super().reset()