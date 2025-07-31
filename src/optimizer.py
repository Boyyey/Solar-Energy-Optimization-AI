"""Optimization module for the Solar Energy Optimization System."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class EnergyOptimizer:
    """Optimizes energy usage by scheduling appliances and battery operations."""
    
    def __init__(self, 
                 battery_capacity_kwh: float = 10.0,
                 max_charge_rate_kw: float = 5.0,
                 battery_efficiency: float = 0.95,
                 initial_soc: float = 0.5):
        """Initialize the energy optimizer.
        
        Args:
            battery_capacity_kwh: Total battery capacity in kWh
            max_charge_rate_kw: Maximum charge/discharge rate in kW
            battery_efficiency: Round-trip efficiency of the battery (0-1)
            initial_soc: Initial state of charge (0-1)
        """
        self.battery_capacity_kwh = battery_capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.battery_efficiency = battery_efficiency
        self.initial_soc = initial_soc
    
    def optimize_schedule(self,
                        demand_forecast: pd.Series,
                        production_forecast: pd.Series,
                        electricity_prices: pd.Series,
                        co2_intensity: pd.Series = None) -> Dict[str, pd.Series]:
        """Optimize energy usage schedule.
        
        Args:
            demand_forecast: Forecasted energy demand in kWh per time period
            production_forecast: Forecasted solar production in kWh per time period
            electricity_prices: Electricity prices per time period (in local currency)
            co2_intensity: CO2 intensity of grid electricity (gCO2/kWh)
            
        Returns:
            Dictionary containing:
                - 'battery_charge': Battery charge schedule (kWh)
                - 'battery_discharge': Battery discharge schedule (kWh)
                - 'grid_import': Energy imported from grid (kWh)
                - 'grid_export': Energy exported to grid (kWh)
                - 'battery_soc': State of charge over time (0-1)
        """
        # Initialize time series
        timesteps = len(demand_forecast)
        time_index = demand_forecast.index
        
        # Initialize result arrays
        battery_charge = np.zeros(timesteps)
        battery_discharge = np.zeros(timesteps)
        grid_import = np.zeros(timesteps)
        grid_export = np.zeros(timesteps)
        battery_soc = np.zeros(timesteps)
        battery_soc[0] = self.initial_soc * self.battery_capacity_kwh  # Start with initial SoC
        
        # If no CO2 intensity data provided, create a constant series of 1.0
        if co2_intensity is None:
            co2_intensity = pd.Series(1.0, index=time_index)
        
        # Simple rule-based optimization
        for t in range(timesteps):
            # Current net production (positive means excess, negative means deficit)
            net_energy = production_forecast.iloc[t] - demand_forecast.iloc[t]
            
            if net_energy > 0:
                # Excess energy - charge battery or export to grid
                if battery_soc[t] < self.battery_capacity_kwh:
                    # Charge battery first
                    charge_amount = min(
                        net_energy,
                        self.max_charge_rate_kw,
                        (self.battery_capacity_kwh - battery_soc[t]) / self.battery_efficiency
                    )
                    battery_charge[t] = charge_amount
                    net_energy -= charge_amount
                
                # If still excess, export to grid
                if net_energy > 0:
                    grid_export[t] = net_energy
            
            elif net_energy < 0:
                # Energy deficit - discharge battery or import from grid
                energy_needed = -net_energy
                
                # Try to discharge battery first
                if battery_soc[t] > 0:
                    discharge_amount = min(
                        energy_needed,
                        self.max_charge_rate_kw,
                        battery_soc[t] * self.battery_efficiency
                    )
                    battery_discharge[t] = discharge_amount
                    energy_needed -= discharge_amount
                
                # If still need energy, import from grid
                if energy_needed > 0:
                    grid_import[t] = energy_needed
            
            # Update battery state of charge for next time step
            if t < timesteps - 1:
                battery_soc[t+1] = (
                    battery_soc[t] 
                    + battery_charge[t] * self.battery_efficiency  # Charging loses some energy
                    - battery_discharge[t] / self.battery_efficiency  # Discharging also loses some
                )
        
        # Create result DataFrame
        result = {
            'battery_charge': pd.Series(battery_charge, index=time_index),
            'battery_discharge': pd.Series(battery_discharge, index=time_index),
            'grid_import': pd.Series(grid_import, index=time_index),
            'grid_export': pd.Series(grid_export, index=time_index),
            'battery_soc': pd.Series(battery_soc, index=time_index) / self.battery_capacity_kwh
        }
        
        return result
    
    def calculate_savings(self, 
                         schedule: Dict[str, pd.Series],
                         electricity_prices: pd.Series) -> Dict[str, float]:
        """Calculate cost savings from the optimized schedule.
        
        Args:
            schedule: Dictionary containing the optimized schedule
            electricity_prices: Electricity prices per time period
            
        Returns:
            Dictionary with savings metrics
        """
        # Calculate costs without optimization (import all demand from grid)
        total_demand = schedule['grid_import'].sum() + schedule['battery_discharge'].sum()
        cost_without_optimization = (total_demand * electricity_prices).sum()
        
        # Calculate costs with optimization
        import_costs = (schedule['grid_import'] * electricity_prices).sum()
        export_income = (schedule['grid_export'] * electricity_prices).sum()
        cost_with_optimization = import_costs - export_income
        
        # Calculate savings
        savings = cost_without_optimization - cost_with_optimization
        savings_percent = (savings / cost_without_optimization * 100) if cost_without_optimization > 0 else 0
        
        return {
            'total_demand_kwh': total_demand,
            'cost_without_optimization': cost_without_optimization,
            'cost_with_optimization': cost_with_optimization,
            'savings': savings,
            'savings_percent': savings_percent,
            'self_consumption_ratio': (1 - (schedule['grid_export'].sum() / 
                                        (schedule['grid_export'].sum() + schedule['battery_charge'].sum())))
                                     if (schedule['grid_export'].sum() + schedule['battery_charge'].sum()) > 0 else 0,
            'autonomy_ratio': (1 - (schedule['grid_import'].sum() / 
                                 (schedule['grid_import'].sum() + schedule['battery_discharge'].sum())))
                            if (schedule['grid_import'].sum() + schedule['battery_discharge'].sum()) > 0 else 0
        }
