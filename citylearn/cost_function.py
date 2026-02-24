from typing import List, Tuple
import numpy as np
import pandas as pd

class CostFunction:
    r"""Cost and energy flexibility functions that may be used to evaluate environment performance."""

    @staticmethod
    def ramping(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of absolute difference in net electric consumption between consecutive time steps.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.

        Returns
        -------
        ramping : List[float]
            Ramping cost.

        Notes
        -----        
        .. math:: 
            \textrm{ramping} = \sum_{i=1}^{n}{\lvert E_i - E_{i-1} \rvert}
            
        Where :math:`E_i` is the :math:`i^{\textrm{th}}` element in `net_electricity_consumption`, :math:`E`, that has a length of :math:`n`.
        """

        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['ramping'] = data['net_electricity_consumption'] - data['net_electricity_consumption'].shift(1)
        data['ramping'] = data['ramping'].abs()
        data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['ramping'].tolist()

    @staticmethod
    def one_minus_load_factor(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Difference between 1 and the load factor i.e., ratio of rolling mean demand 
        to rolling peak demand over a specified period.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 730
            Period window/time steps.

        Returns
        -------
        1 - load_factor : List[float]
            1 - load factor cost.
        """

        window = 730 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['group'] = (data.index/window).astype(int)
        data = data.groupby(['group'])[['net_electricity_consumption']].agg(['mean','max'])
        data['load_factor'] = 1 - (data[('net_electricity_consumption','mean')]/data[('net_electricity_consumption','max')])
        data['load_factor'] = data['load_factor'].rolling(window=data.shape[0],min_periods=1).mean()
        
        return data['load_factor'].tolist()

    @staticmethod
    def peak(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Net electricity consumption peak.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 24
            Period window/time steps to find peaks.
            
        Returns
        -------
        peak : List[float]
            Average daily peak cost.
        """

        window = 24 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['group'] = (data.index/window).astype(int)
        data = data.groupby(['group'])[['net_electricity_consumption']].max()
        data['net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).mean()
        
        return data['net_electricity_consumption'].tolist()

    @staticmethod
    def electricity_consumption(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of positive electricity consumption.

        It is the sum of electricity that is consumed from the grid.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        electricity_consumption : List[float]
            Electricity consumption cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['electricity_consumption'].tolist()

    @staticmethod
    def zero_net_energy(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of net electricity consumption.

        It is the net sum of electricty that is consumed from the grid and self-generated from renenewable sources.
        This calculation of zero net energy does not consider TDV and all time steps are weighted equally.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        zero_net_energy : List[float]
            Zero net energy cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption)})
        data['zero_net_energy'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['zero_net_energy'].tolist()

    @staticmethod
    def carbon_emissions(carbon_emissions: List[float]) -> List[float]:
        r"""Rolling sum of carbon emissions.

        Parameters
        ----------
        carbon_emissions : List[float]
            Carbon emissions time series.
            
        Returns
        -------
        carbon_emissions : List[float]
            Carbon emissions cost.
        """

        data = pd.DataFrame({'carbon_emissions':np.array(carbon_emissions).clip(min=0)})
        data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['carbon_emissions'].tolist()

    @staticmethod
    def cost(cost: List[float]) -> List[float]:
        r"""Rolling sum of electricity monetary cost.

        Parameters
        ----------
        cost : List[float]
            Cost time series.
            
        Returns
        -------
        cost : List[float]
            Cost of electricity.
        """

        data = pd.DataFrame({'cost':np.array(cost).clip(min=0)})
        data['cost'] = data['cost'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['cost'].tolist()

    @staticmethod
    def quadratic(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of net electricity consumption raised to the power of 2.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        quadratic : List[float]
            Quadratic cost.

        Notes
        -----
        Net electricity consumption values are clipped at a minimum of 0 before calculating the quadratic cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['quadratic'] = data['net_electricity_consumption']**2
        data['quadratic'] = data['quadratic'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['quadratic'].tolist()
    
    @staticmethod
    def discomfort(indoor_dry_bulb_temperature: List[float], dry_bulb_temperature_set_point: List[float], band: float = None, occupant_count: List[int] = None) -> Tuple[list]:
        r"""Rolling percentage of discomfort (total, too cold, and too hot) time steps as well as rolling minimum, maximum and average temperature delta.

        Parameters
        ----------
        indoor_dry_bulb_temperature: List[float]
            Average building dry bulb temperature time series.
        dry_bulb_temperature_set_point: List[float]
            Building thermostat setpoint time series.
        band: float, default = 2.0
            Comfort band above and below dry_bulb_temperature_set_point beyond 
            which occupant is assumed to be uncomfortable.
        occupant_cunt: List[float], optional
            Occupant count time series. If provided, the comfort cost is 
            evaluated for associated_EV time steps only.
            
        Returns
        -------
        discomfort: List[float]
            Rolling proportion of associated_EV timesteps where the condition
            (dry_bulb_temperature_set_point - band) <= indoor_dry_bulb_temperature <= (dry_bulb_temperature_set_point + band) is not met.
        discomfort_too_cold: List[float]
            Rolling proportion of associated_EV timesteps where the condition indoor_dry_bulb_temperature < (dry_bulb_temperature_set_point - band) is met.
        discomfort_too_hot: List[float]
            Rolling proportion of associated_EV timesteps where the condition indoor_dry_bulb_temperature > (dry_bulb_temperature_set_point + band) is met.
        discomfort_delta_minimum: List[float]
            Rolling minimum of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        discomfort_delta_maximum: List[float]
            Rolling maximum of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        discomfort_delta_average: List[float]
            Rolling average of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        """

        band = 2.0 if band is None else band

        # unmet hours
        data = pd.DataFrame({
            'indoor_dry_bulb_temperature': indoor_dry_bulb_temperature, 
            'dry_bulb_temperature_set_point': dry_bulb_temperature_set_point,
            'occupant_count': [1]*len(indoor_dry_bulb_temperature) if occupant_count is None else occupant_count
        })
        occupied_time_step_count = data[data['occupant_count'] > 0.0].shape[0]
        data['delta'] = data['indoor_dry_bulb_temperature'] - data['dry_bulb_temperature_set_point']
        data.loc[data['occupant_count'] == 0.0, 'delta'] = 0.0
        data['discomfort'] = 0
        data.loc[data['delta'].abs() > band, 'discomfort'] = 1
        data['discomfort'] = data['discomfort'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # too cold
        data['discomfort_too_cold'] = 0
        data.loc[data['delta'] < -band, 'discomfort_too_cold'] = 1
        data['discomfort_too_cold'] = data['discomfort_too_cold'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # too hot
        data['discomfort_too_hot'] = 0
        data.loc[data['delta'] > band, 'discomfort_too_hot'] = 1
        data['discomfort_too_hot'] = data['discomfort_too_hot'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # minimum delta
        data['discomfort_delta_minimum'] = data['delta'].rolling(window=data.shape[0],min_periods=1).min()

        # maximum delta
        data['discomfort_delta_maximum'] = data['delta'].rolling(window=data.shape[0],min_periods=1).max()

        # average delta
        data['discomfort_delta_average'] = data['delta'].rolling(window=data.shape[0],min_periods=1).mean()

        return (
            data['discomfort'].tolist(),
            data['discomfort_too_cold'].tolist(),
            data['discomfort_too_hot'].tolist(),
            data['discomfort_delta_minimum'].tolist(),
            data['discomfort_delta_maximum'].tolist(),
            data['discomfort_delta_average'].tolist()
        )
        
    @staticmethod
    def secondary_storage_efficiency(secondary_storage_energy_balance: List[float], 
                                   net_electricity_consumption: List[float]) -> float:
        """Calculate secondary storage efficiency in reducing grid interaction.
        
        Parameters
        ----------
        secondary_storage_energy_balance : List[float]
            Secondary storage energy balance time series.
        net_electricity_consumption : List[float]
            Net electricity consumption time series.
            
        Returns
        -------
        efficiency : float
            Secondary storage efficiency ratio (0-1).
        """
        if not secondary_storage_energy_balance or not net_electricity_consumption:
            return 0.0
            
        total_storage_energy = sum(abs(x) for x in secondary_storage_energy_balance)
        total_consumption = sum(abs(x) for x in net_electricity_consumption)
        
        if total_consumption == 0:
            return 0.0
            
        return min(total_storage_energy / total_consumption, 1.0)
    
    @staticmethod
    def energy_sharing_ratio(secondary_storage_energy_balance: List[float],
                           building_surpluses: List[List[float]]) -> float:
        """Calculate the ratio of energy successfully shared through secondary storage.
        
        Parameters
        ----------
        secondary_storage_energy_balance : List[float]
            Secondary storage energy balance time series.
        building_surpluses : List[List[float]]
            Surplus energy for each building over time.
            
        Returns
        -------
        sharing_ratio : float
            Ratio of energy shared vs total surplus (0-1).
        """
        if not secondary_storage_energy_balance or not building_surpluses:
            return 0.0
            
        total_shared = sum(abs(x) for x in secondary_storage_energy_balance)
        total_surplus = sum(sum(abs(x) for x in building if x < 0) 
                          for building in building_surpluses)
        
        if total_surplus == 0:
            return 0.0
            
        return min(total_shared / total_surplus, 1.0)
    
    @staticmethod
    def secondary_storage_utilization(secondary_storage_soc: List[float]) -> dict:
        """Calculate secondary storage utilization metrics.
        
        Parameters
        ----------
        secondary_storage_soc : List[float]
            Secondary storage state of charge time series.
            
        Returns
        -------
        utilization_metrics : dict
            Dictionary containing utilization metrics.
        """
        if not secondary_storage_soc:
            return {
                'average_soc': 0.0,
                'soc_variance': 0.0,
                'utilization_rate': 0.0,
                'cycling_frequency': 0.0
            }
        
        data = pd.DataFrame({'soc': secondary_storage_soc})
        
        # Calculate metrics
        average_soc = data['soc'].mean()
        soc_variance = data['soc'].var()
        
        # Utilization rate: how much of the capacity range is used
        soc_range = data['soc'].max() - data['soc'].min()
        utilization_rate = soc_range
        
        # Cycling frequency: number of charge/discharge cycles
        soc_diff = data['soc'].diff().fillna(0)
        direction_changes = (soc_diff * soc_diff.shift(1) < 0).sum()
        cycling_frequency = direction_changes / len(data) if len(data) > 0 else 0.0
        
        return {
            'average_soc': average_soc,
            'soc_variance': soc_variance,
            'utilization_rate': utilization_rate,
            'cycling_frequency': cycling_frequency
        }
    
    @staticmethod
    def grid_interaction_reduction(net_consumption_with_storage: List[float],
                                 net_consumption_without_storage: List[float]) -> float:
        """Calculate grid interaction reduction due to secondary storage.
        
        Parameters
        ----------
        net_consumption_with_storage : List[float]
            Net electricity consumption with secondary storage.
        net_consumption_without_storage : List[float]
            Net electricity consumption without secondary storage.
            
        Returns
        -------
        reduction_ratio : float
            Grid interaction reduction ratio (0-1).
        """
        if not net_consumption_with_storage or not net_consumption_without_storage:
            return 0.0
            
        total_with = sum(abs(x) for x in net_consumption_with_storage)
        total_without = sum(abs(x) for x in net_consumption_without_storage)
        
        if total_without == 0:
            return 0.0
            
        reduction = max(0, total_without - total_with)
        return reduction / total_without
    
    @staticmethod
    def secondary_storage_kpis(secondary_storage_soc: List[float],
                             secondary_storage_energy_balance: List[float],
                             net_electricity_consumption: List[float],
                             building_surpluses: List[List[float]] = None,
                             building_net_consumptions: List[List[float]] = None,
                             building_ss_requests: List[List[float]] = None,
                             electricity_prices: List[float] = None,
                             solar_generation: List[float] = None,
                             battery_capacity: float = 200.0,
                             battery_efficiency: float = 0.95) -> dict:
        """Calculate comprehensive secondary storage KPIs.
        
        Parameters
        ----------
        secondary_storage_soc : List[float]
            Secondary storage state of charge time series (fraction 0-1).
        secondary_storage_energy_balance : List[float]
            Secondary storage energy balance time series (positive=charge, negative=discharge).
        net_electricity_consumption : List[float]
            District-level net electricity consumption time series.
        building_surpluses : List[List[float]], optional
            Surplus energy for each building over time.
        building_net_consumptions : List[List[float]], optional
            Per-building net consumption time series. Each inner list is one building.
        building_ss_requests : List[List[float]], optional
            Per-building secondary storage action requests. Each inner list is one building.
        electricity_prices : List[float], optional
            Electricity pricing time series.
        solar_generation : List[float], optional
            Total solar generation time series.
        battery_capacity : float
            Battery capacity in kWh.
        battery_efficiency : float
            Battery round-trip efficiency.
            
        Returns
        -------
        kpis : dict
            Comprehensive KPI dictionary.
        """
        kpis = {}
        
        # ---- Basic utilization metrics ----
        utilization_metrics = CostFunction.secondary_storage_utilization(secondary_storage_soc)
        kpis.update(utilization_metrics)
        
        # ---- Efficiency metrics ----
        kpis['storage_efficiency'] = CostFunction.secondary_storage_efficiency(
            secondary_storage_energy_balance, net_electricity_consumption
        )
        
        # ---- Energy sharing metrics ----
        if building_surpluses:
            kpis['energy_sharing_ratio'] = CostFunction.energy_sharing_ratio(
                secondary_storage_energy_balance, building_surpluses
            )
        
        # ---- Energy flow metrics ----
        if secondary_storage_energy_balance:
            eb = secondary_storage_energy_balance
            total_energy_flow = sum(abs(x) for x in eb)
            charging_energy = sum(x for x in eb if x > 0)
            discharging_energy = sum(abs(x) for x in eb if x < 0)
            
            kpis['total_energy_flow'] = total_energy_flow
            kpis['charging_energy'] = charging_energy
            kpis['discharging_energy'] = discharging_energy
            kpis['charge_discharge_ratio'] = (charging_energy / discharging_energy 
                                            if discharging_energy > 0 else 0.0)
            
            # Round-trip efficiency: energy_out / energy_in
            kpis['round_trip_efficiency'] = (discharging_energy / charging_energy
                                            if charging_energy > 0 else 0.0)
            
            # Idle time: fraction of steps with no significant action
            idle_steps = sum(1 for x in eb if abs(x) < 0.1)
            kpis['idle_fraction'] = idle_steps / len(eb) if len(eb) > 0 else 1.0
            
            # Average charge/discharge power
            charge_steps = [x for x in eb if x > 0.1]
            discharge_steps = [abs(x) for x in eb if x < -0.1]
            kpis['avg_charging_power'] = np.mean(charge_steps) if charge_steps else 0.0
            kpis['avg_discharging_power'] = np.mean(discharge_steps) if discharge_steps else 0.0
        
        # ---- Performance indicators ----
        if net_electricity_consumption:
            nc = net_electricity_consumption
            peak_consumption = max(abs(x) for x in nc)
            average_consumption = np.mean([abs(x) for x in nc])
            kpis['peak_consumption'] = peak_consumption
            kpis['average_consumption'] = average_consumption
            kpis['peak_to_average_ratio'] = (peak_consumption / average_consumption 
                                            if average_consumption > 0 else 0.0)
        
        # ---- Surplus capture rate ----
        # What fraction of total surplus was captured into SS?
        if building_net_consumptions and secondary_storage_energy_balance:
            T = min(len(secondary_storage_energy_balance), 
                    min(len(bc) for bc in building_net_consumptions))
            total_surplus = 0.0
            total_deficit = 0.0
            for t in range(T):
                for bc in building_net_consumptions:
                    if bc[t] < 0:
                        total_surplus += abs(bc[t])
                    else:
                        total_deficit += bc[t]
            
            charged = sum(x for x in secondary_storage_energy_balance[:T] if x > 0)
            discharged = sum(abs(x) for x in secondary_storage_energy_balance[:T] if x < 0)
            
            kpis['surplus_capture_rate'] = min(charged / total_surplus, 1.0) if total_surplus > 0 else 0.0
            kpis['deficit_coverage_rate'] = min(discharged / total_deficit, 1.0) if total_deficit > 0 else 0.0
        
        # ---- Action correctness ----
        # Fraction of time steps where the SS action direction matches surplus/deficit
        if building_net_consumptions and building_ss_requests:
            n_buildings = len(building_net_consumptions)
            T = min(len(building_net_consumptions[0]), len(building_ss_requests[0]))
            correct = 0
            total_actionable = 0
            
            for t in range(T):
                for bi in range(n_buildings):
                    net = building_net_consumptions[bi][t] if t < len(building_net_consumptions[bi]) else 0
                    act = building_ss_requests[bi][t] if t < len(building_ss_requests[bi]) else 0
                    if abs(net) > 0.5:  # only count when there's meaningful surplus/deficit
                        total_actionable += 1
                        if net < -0.5 and act > 0.02:  # surplus + charge = correct
                            correct += 1
                        elif net > 0.5 and act < -0.02:  # deficit + discharge = correct
                            correct += 1
            
            kpis['action_correctness'] = correct / total_actionable if total_actionable > 0 else 0.0
            kpis['actionable_steps'] = total_actionable
            kpis['correct_action_steps'] = correct
        
        # ---- SOC health ratio ----
        # Fraction of time SOC is in healthy range [0.1, 0.9]
        if secondary_storage_soc:
            healthy = sum(1 for s in secondary_storage_soc if 0.1 <= s <= 0.9)
            kpis['soc_health_ratio'] = healthy / len(secondary_storage_soc)
            
            # SOC extremes
            kpis['soc_min'] = min(secondary_storage_soc)
            kpis['soc_max'] = max(secondary_storage_soc)
            kpis['soc_final'] = secondary_storage_soc[-1]
            
            # Time at extreme SOC
            empty_steps = sum(1 for s in secondary_storage_soc if s < 0.05)
            full_steps = sum(1 for s in secondary_storage_soc if s > 0.95)
            kpis['time_at_empty'] = empty_steps / len(secondary_storage_soc)
            kpis['time_at_full'] = full_steps / len(secondary_storage_soc)
        
        # ---- Pricing-aware metrics ----
        if electricity_prices and secondary_storage_energy_balance:
            T = min(len(electricity_prices), len(secondary_storage_energy_balance))
            charge_cost = 0.0
            discharge_revenue = 0.0
            for t in range(T):
                price = electricity_prices[t]
                eb_val = secondary_storage_energy_balance[t]
                if eb_val > 0:  # charging
                    charge_cost += price * eb_val
                elif eb_val < 0:  # discharging
                    discharge_revenue += price * abs(eb_val)
            
            kpis['charge_cost'] = charge_cost
            kpis['discharge_revenue'] = discharge_revenue
            kpis['arbitrage_profit'] = discharge_revenue - charge_cost
            kpis['avg_charge_price'] = (charge_cost / kpis.get('charging_energy', 1.0)
                                       if kpis.get('charging_energy', 0) > 0 else 0.0)
            kpis['avg_discharge_price'] = (discharge_revenue / kpis.get('discharging_energy', 1.0)
                                          if kpis.get('discharging_energy', 0) > 0 else 0.0)
        
        # ---- Solar utilization metrics ----
        if solar_generation and secondary_storage_energy_balance:
            T = min(len(solar_generation), len(secondary_storage_energy_balance))
            solar_to_ss = 0.0
            total_solar = sum(solar_generation[:T])
            for t in range(T):
                if solar_generation[t] > 0 and secondary_storage_energy_balance[t] > 0:
                    solar_to_ss += min(solar_generation[t], secondary_storage_energy_balance[t])
            kpis['solar_to_ss_ratio'] = solar_to_ss / total_solar if total_solar > 0 else 0.0
            kpis['total_solar_generation'] = total_solar
        
        # ---- Demand response effectiveness ----
        # How much does SS reduce peak demand compared to average?
        if net_electricity_consumption and secondary_storage_energy_balance:
            nc = net_electricity_consumption
            T = min(len(nc), len(secondary_storage_energy_balance))
            # Estimate what consumption would be without SS
            nc_without_ss = [
                nc[t] + secondary_storage_energy_balance[t] for t in range(T)
            ]
            peak_with = max(abs(x) for x in nc[:T]) if T > 0 else 0.0
            peak_without = max(abs(x) for x in nc_without_ss) if T > 0 else 0.0
            kpis['peak_reduction_ratio'] = (
                (peak_without - peak_with) / peak_without
                if peak_without > 0 else 0.0
            )
        
        # ---- Self-sufficiency improvement ----
        # Fraction of grid imports avoided by SS discharge
        if net_electricity_consumption and secondary_storage_energy_balance:
            T = min(len(net_electricity_consumption), len(secondary_storage_energy_balance))
            grid_imports_with = sum(
                max(net_electricity_consumption[t], 0) for t in range(T)
            )
            grid_imports_without = sum(
                max(net_electricity_consumption[t] + secondary_storage_energy_balance[t], 0)
                for t in range(T)
            )
            kpis['self_sufficiency_improvement'] = (
                (grid_imports_without - grid_imports_with) / grid_imports_without
                if grid_imports_without > 0 else 0.0
            )
        
        # ---- Energy waste ratio ----
        # Energy lost due to battery inefficiency
        if secondary_storage_energy_balance:
            eb = secondary_storage_energy_balance
            total_in = sum(x for x in eb if x > 0)
            total_out = sum(abs(x) for x in eb if x < 0)
            energy_loss = total_in - total_out
            kpis['energy_waste_kwh'] = max(energy_loss, 0.0)
            kpis['energy_waste_ratio'] = (
                max(energy_loss, 0.0) / total_in if total_in > 0 else 0.0
            )
        
        # ---- Response rate ----
        # Fraction of actionable steps where SS actually responded
        if building_net_consumptions and building_ss_requests:
            n_buildings = len(building_net_consumptions)
            T = min(len(building_net_consumptions[0]), len(building_ss_requests[0]))
            responded = 0
            actionable = 0
            for t in range(T):
                for bi in range(n_buildings):
                    net = building_net_consumptions[bi][t] if t < len(building_net_consumptions[bi]) else 0
                    act = building_ss_requests[bi][t] if t < len(building_ss_requests[bi]) else 0
                    if abs(net) > 0.5:
                        actionable += 1
                        if abs(act) > 0.02:
                            responded += 1
            kpis['response_rate'] = responded / actionable if actionable > 0 else 0.0
        
        # ---- Temporal arbitrage score ----
        # Ratio of avg discharge price to avg charge price (>1 = profitable)
        if electricity_prices and secondary_storage_energy_balance:
            avg_cp = kpis.get('avg_charge_price', 0.0)
            avg_dp = kpis.get('avg_discharge_price', 0.0)
            kpis['temporal_arbitrage_score'] = (
                avg_dp / avg_cp if avg_cp > 0 else 0.0
            )
        
        # ---- Cycling depth ----
        # Average SOC swing per charge/discharge cycle
        if secondary_storage_soc and len(secondary_storage_soc) > 2:
            soc_arr = np.array(secondary_storage_soc)
            diffs = np.diff(soc_arr)
            # Find direction changes (sign flips)
            signs = np.sign(diffs)
            sign_changes = np.where(signs[:-1] != signs[1:])[0]
            if len(sign_changes) > 1:
                cycle_depths = []
                for i in range(len(sign_changes) - 1):
                    start_idx = sign_changes[i]
                    end_idx = sign_changes[i + 1]
                    cycle_range = abs(soc_arr[start_idx] - soc_arr[end_idx])
                    cycle_depths.append(cycle_range)
                kpis['avg_cycling_depth'] = float(np.mean(cycle_depths)) if cycle_depths else 0.0
                kpis['max_cycling_depth'] = float(np.max(cycle_depths)) if cycle_depths else 0.0
                kpis['num_full_cycles'] = len(cycle_depths)
            else:
                kpis['avg_cycling_depth'] = 0.0
                kpis['max_cycling_depth'] = 0.0
                kpis['num_full_cycles'] = 0
        
        # ---- SS constraint violations ----
        if secondary_storage_soc and building_ss_requests:
            n_buildings = len(building_ss_requests)
            T = min(len(secondary_storage_soc), min(len(r) for r in building_ss_requests))
            charge_when_full = 0
            discharge_when_empty = 0
            for t in range(T):
                soc = secondary_storage_soc[t]
                avg_req = sum(building_ss_requests[bi][t] for bi in range(n_buildings)) / n_buildings
                if soc > 0.95 and avg_req > 0.05:
                    charge_when_full += 1
                if soc < 0.05 and avg_req < -0.05:
                    discharge_when_empty += 1
            kpis['ss_charge_when_full_count'] = charge_when_full
            kpis['ss_discharge_when_empty_count'] = discharge_when_empty
            kpis['ss_constraint_violation_rate'] = (
                (charge_when_full + discharge_when_empty) / T if T > 0 else 0.0
            )
        
        return kpis