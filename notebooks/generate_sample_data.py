"""
Sample Data Generation Script for Solar Energy Optimization System

This script generates sample data for demonstration purposes.
It creates realistic time series data for energy demand, solar production, and weather.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ensure data directory exists
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def generate_sample_data(start_date=None, days=365):
    """Generate sample data for the Solar Energy Optimization System.
    
    Args:
        start_date: Start date for the data (default: one year ago from today)
        days: Number of days of data to generate
        
    Returns:
        Dictionary containing DataFrames for demand, solar, and weather data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Create hourly timestamps
    timestamps = pd.date_range(
        start=start_date,
        periods=days*24,
        freq='H'
    )
    
    # Generate seasonal patterns
    day_of_year = timestamps.dayofyear
    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek
    
    # Base patterns
    seasonal = 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Annual seasonality
    daily = 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)      # Daily pattern
    weekly = 0.1 * (day_of_week >= 5)                              # Weekend effect
    
    # 1. Generate demand data (kWh)
    base_demand = 5.0  # Average base demand in kW
    
    # Add randomness and patterns
    demand = base_demand * (1 + seasonal + daily + weekly + 0.2 * np.random.randn(len(timestamps)))
    demand = np.maximum(0.5, demand)  # Ensure positive demand
    
    # Create demand DataFrame
    demand_df = pd.DataFrame({
        'timestamp': timestamps,
        'demand_kwh': demand,
        'temperature': np.nan,  # Will be filled from weather data
        'is_holiday': (timestamps.month == 1) & (timestamps.day == 1)  # Example holiday
    })
    
    # 2. Generate solar production data (kWh)
    # Solar potential (0 at night, peaks at noon)
    # Convert to numpy array first to allow modification
    solar_potential = np.sin(np.pi * (hour_of_day.values - 6) / 12)
    solar_potential = np.where(solar_potential < 0, 0, solar_potential)
    
    # Seasonal adjustment (more sun in summer)
    solar_seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # System capacity (kWp)
    system_capacity = 8.0  # 8 kWp system
    
    # Generate solar production
    solar = system_capacity * solar_potential * solar_seasonal * (0.9 + 0.2 * np.random.rand(len(timestamps)))
    
    # Create solar DataFrame
    solar_df = pd.DataFrame({
        'timestamp': timestamps,
        'solar_production_kwh': solar,
        'solar_irradiance': 1000 * solar_potential * (0.8 + 0.4 * np.random.rand(len(timestamps)))
    })
    
    # 3. Generate weather data
    # Temperature (seasonal pattern with daily variation)
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Annual cycle
    daily_temp_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)  # Warmer in the afternoon
    temperature = base_temp + daily_temp_variation + 3 * np.random.randn(len(timestamps))
    
    # Cloud cover (0-100%)
    cloud_cover = 30 + 20 * np.sin(2 * np.pi * day_of_year / 20)  # ~3-week cycles
    cloud_cover = np.clip(cloud_cover + 20 * np.random.randn(len(timestamps)), 0, 100)
    
    # Create weather DataFrame
    weather_df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'cloud_cover': cloud_cover,
        'humidity': np.clip(60 + 30 * np.sin(2 * np.pi * hour_of_day / 24) + 10 * np.random.randn(len(timestamps)), 20, 100),
        'wind_speed': np.maximum(0, 5 + 3 * np.random.randn(len(timestamps))),
        'precipitation': np.random.exponential(0.2, len(timestamps)) * (np.random.rand(len(timestamps)) < 0.1)  # 10% chance of rain
    })
    
    # Add temperature to demand data (for demand forecasting)
    demand_df['temperature'] = weather_df['temperature']
    
    return {
        'demand': demand_df,
        'solar': solar_df,
        'weather': weather_df
    }

def save_sample_data(data, output_dir=DATA_DIR):
    """Save sample data to CSV files.
    
    Args:
        data: Dictionary containing DataFrames for demand, solar, and weather
        output_dir: Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save each DataFrame to a separate CSV file
    data['demand'].to_csv(output_dir / 'demand_data.csv', index=False)
    data['solar'].to_csv(output_dir / 'solar_data.csv', index=False)
    data['weather'].to_csv(output_dir / 'weather_data.csv', index=False)
    
    print(f"Sample data saved to {output_dir.absolute()}")
    print(f"- demand_data.csv: {len(data['demand'])} rows")
    print(f"- solar_data.csv: {len(data['solar'])} rows")
    print(f"- weather_data.csv: {len(data['weather'])} rows")

if __name__ == "__main__":
    print("Generating sample data for Solar Energy Optimization System...")
    
    # Generate one year of hourly data
    sample_data = generate_sample_data(days=365)
    
    # Save the data to CSV files
    save_sample_data(sample_data)
    
    print("\nSample data generation complete!")
