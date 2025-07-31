"""
Generate example CSV files for Bushehr, Iran with realistic data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demand_data():
    """Generate realistic demand data for Bushehr."""
    np.random.seed(42)  # For reproducibility
    
    # Create a date range for one week with hourly frequency
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)  # 7 days total
    date_range = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=23, minute=0, second=0, microsecond=0),
        freq='H'
    )
    
    # Base demand patterns
    base_demand = 5.0  # Base demand in kW
    
    # Add daily and weekly patterns
    daily_pattern = 0.4 * np.sin(2 * np.pi * (date_range.hour - 6) / 24)  # Peak in the afternoon
    weekly_pattern = 0.1 * (date_range.dayofweek >= 5)  # Higher on weekends
    
    # Add some noise
    noise = 0.2 * np.random.randn(len(date_range))
    
    # Combine all components
    demand = base_demand * (1 + daily_pattern + weekly_pattern + noise)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'demand_kwh': np.maximum(0.5, demand)  # Ensure positive demand
    })
    
    # Save to CSV
    df.to_csv('data/demand_data.csv', index=False)
    return df

def generate_solar_data():
    """Generate realistic solar production data for Bushehr."""
    np.random.seed(43)  # Different seed for different random values
    
    # Create a date range for one week with hourly frequency
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)  # 7 days total
    date_range = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=23, minute=0, second=0, microsecond=0),
        freq='H'
    )
    
    # Solar generation pattern (sinusoidal during daylight hours)
    solar_gen = np.zeros(len(date_range))
    for i, dt in enumerate(date_range):
        hour = dt.hour
        if 5 <= hour <= 19:  # Daylight hours
            # Parabolic curve peaking at solar noon
            t = (hour - 5) / 14  # Normalized time (0-1)
            solar_gen[i] = 8.0 * (4 * t * (1 - t)) ** 2  # Peak at 8kW
    
    # Add some noise and cloud cover variation
    noise = 0.2 * np.random.randn(len(date_range))
    cloud_effect = 1.0 - 0.3 * (np.random.random(len(date_range)) < 0.2)  # 20% chance of 30% reduction
    
    # Combine effects
    solar_gen = np.maximum(0, solar_gen * (1 + noise) * cloud_effect)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'solar_production_kwh': solar_gen
    })
    
    # Save to CSV
    df.to_csv('data/solar_data.csv', index=False)
    return df

def generate_weather_data():
    """Generate realistic weather data for Bushehr."""
    np.random.seed(44)  # Different seed for different random values
    
    # Create a date range for one week with hourly frequency
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)  # 7 days total
    date_range = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=23, minute=0, second=0, microsecond=0),
        freq='H'
    )
    
    # Base temperature (in Celsius)
    base_temp = 30.0  # Nighttime low
    daily_swing = 20.0  # Day-night temperature difference
    
    # Temperature follows a daily pattern
    temp = []
    for dt in date_range:
        # Normalized hour (0-1) for the day
        t = (dt.hour + dt.minute/60) / 24
        # Create a daily temperature curve (sinusoidal)
        daily_temp = base_temp + daily_swing * np.sin(np.pi * (t - 0.25))
        temp.append(daily_temp)
    
    temp = np.array(temp)
    
    # Add some daily variation (heat waves, etc.)
    day_variation = 5.0 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24*7))  # Weekly pattern
    temp += day_variation
    
    # Cap at 51°C as requested
    temp = np.minimum(51.0, temp)
    
    # Add some noise
    temp += 2.0 * np.random.randn(len(temp))
    
    # Cloud cover (0-1)
    cloud_cover = 0.1 + 0.3 * np.random.random(len(date_range))  # Mostly clear skies
    
    # Wind speed (m/s)
    wind_speed = 2.0 + 3.0 * np.random.random(len(date_range))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'temperature': temp,
        'cloud_cover': cloud_cover,
        'wind_speed': wind_speed
    })
    
    # Save to CSV
    df.to_csv('data/weather_data.csv', index=False)
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate all data
    print("Generating demand data...")
    demand_df = generate_demand_data()
    
    print("Generating solar production data...")
    solar_df = generate_solar_data()
    
    print("Generating weather data...")
    weather_df = generate_weather_data()
    
    print("\nSample data generated successfully!")
    print(f"- Demand data shape: {demand_df.shape}")
    print(f"- Solar data shape: {solar_df.shape}")
    print(f"- Weather data shape: {weather_df.shape}")
    print("\nFiles saved to the 'data' directory.")
