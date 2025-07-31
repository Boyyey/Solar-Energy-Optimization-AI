"""Utility functions for the Solar Energy Optimization System."""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    default_config = {
        'battery': {
            'capacity_kwh': 10.0,
            'max_charge_rate_kw': 5.0,
            'efficiency': 0.95,
            'initial_soc': 0.5
        },
        'forecasting': {
            'demand_model_type': 'prophet',  # 'prophet' or 'xgboost'
            'solar_model_type': 'prophet',   # 'prophet' or 'xgboost'
            'forecast_horizon_hours': 24
        },
        'api_keys': {
            'openweathermap': None,
            'weatherapi': None
        },
        'data': {
            'demand_data_path': 'data/demand_data.csv',
            'solar_data_path': 'data/solar_data.csv',
            'weather_data_path': 'data/weather_data.csv'
        }
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Merge with default config to ensure all keys exist
        return {**default_config, **config}
    except FileNotFoundError:
        # If config file doesn't exist, create it with default values
        save_config(default_config, config_path)
        return default_config


def save_config(config: Dict[str, Any], config_path: str = 'config.json') -> None:
    """Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def ensure_data_directories() -> None:
    """Ensure that required data directories exist."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """Generate sample data for demonstration purposes.
    
    Returns:
        Dictionary containing sample data for demand, solar production, and weather
    """
    # Create a date range for the next 24 hours
    now = pd.Timestamp.now().normalize()
    hours = pd.date_range(now, now + pd.Timedelta(hours=23), freq='H')
    
    # Generate sample demand (higher during evening)
    demand = 1.5 + np.sin(np.linspace(0, 4*np.pi, 24)) * 0.8
    demand = np.maximum(0.5, demand)  # Minimum 0.5 kW
    
    # Generate sample solar production (0 at night, peaks at noon)
    solar = 3 * np.sin(np.linspace(0, np.pi, 24)) ** 2
    
    # Generate sample electricity prices (higher during peak hours)
    prices = 0.15 + 0.1 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24))
    
    # Generate sample temperature (colder at night)
    temp = 20 + 10 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24))
    
    # Generate sample cloud cover
    cloud_cover = 30 + 40 * np.sin(np.linspace(0, 2*np.pi, 24)) ** 2
    
    # Create DataFrames
    demand_df = pd.DataFrame({
        'timestamp': hours,
        'demand_kwh': demand
    })
    
    solar_df = pd.DataFrame({
        'timestamp': hours,
        'solar_production_kwh': solar
    })
    
    weather_df = pd.DataFrame({
        'timestamp': hours,
        'temperature': temp,
        'cloud_cover': cloud_cover,
        'electricity_price': prices
    })
    
    return {
        'demand': demand_df,
        'solar': solar_df,
        'weather': weather_df
    }


def save_sample_data() -> None:
    """Save sample data to CSV files for demonstration."""
    data = load_sample_data()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save data to CSV files
    data['demand'].to_csv('data/demand_data.csv', index=False)
    data['solar'].to_csv('data/solar_data.csv', index=False)
    data['weather'].to_csv('data/weather_data.csv', index=False)


def calculate_savings_metrics(schedule: Dict[str, pd.Series], 
                            electricity_prices: pd.Series) -> Dict[str, float]:
    """Calculate various savings metrics from the optimization results.
    
    Args:
        schedule: Dictionary containing the optimized schedule
        electricity_prices: Electricity prices per time period
        
    Returns:
        Dictionary with various savings metrics
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
    
    # Calculate self-consumption ratio
    total_solar = schedule['grid_export'].sum() + schedule['battery_charge'].sum()
    self_consumption_ratio = (1 - (schedule['grid_export'].sum() / total_solar)) if total_solar > 0 else 0
    
    # Calculate autonomy ratio
    total_consumption = schedule['grid_import'].sum() + schedule['battery_discharge'].sum()
    autonomy_ratio = (1 - (schedule['grid_import'].sum() / total_consumption)) if total_consumption > 0 else 0
    
    return {
        'total_demand_kwh': total_demand,
        'cost_without_optimization': cost_without_optimization,
        'cost_with_optimization': cost_with_optimization,
        'savings': savings,
        'savings_percent': savings_percent,
        'self_consumption_ratio': self_consumption_ratio,
        'autonomy_ratio': autonomy_ratio,
        'battery_cycles': schedule['battery_charge'].sum() / 10.0  # Assuming 10kWh battery
    }
